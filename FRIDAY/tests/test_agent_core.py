import unittest

from agent.models import AgentTask, StepResult, TaskStatus
from agent.policy import ActionPolicy
from agent.planner import TaskPlanner
from agent.orchestrator import AgentOrchestrator
from config import load_config
from intelligence.response_parser import ActionRequest


class ExecutionReport:
    def __init__(self, success, message="", error=None, actions_completed=0):
        self.success = success
        self.message = message
        self.error = error
        self.actions_completed = actions_completed


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def query(self, system_prompt, user_message, use_cache=False):
        self.calls.append(user_message)
        return self.responses.pop(0)


class FakeExecutor:
    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def execute_chain(self, response, user_input):
        self.calls.append(response.actions[0].action)
        return self.reports.pop(0)


class AgentCoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_policy_requires_confirmation_for_whatsapp(self):
        policy = ActionPolicy(self.cfg)
        decision = policy.evaluate(ActionRequest(action="send_whatsapp", name="Aman", message="hello"))
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_policy_allows_safe_open_app(self):
        policy = ActionPolicy(self.cfg)
        decision = policy.evaluate(ActionRequest(action="open_app", value="chrome"))
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.requires_confirmation)

    def test_planner_parses_structured_plan(self):
        raw = '''{
          "actions": [
            {"action": "open_app", "value": "chrome", "description": "Open Chrome"},
            {"action": "open_website", "value": "https://example.com", "description": "Open the site"}
          ],
          "meta": {"intent": "task", "confidence": 0.95, "plan_complexity": "medium"}
        }'''
        planner = TaskPlanner(self.cfg, FakeLLM([raw]))
        result = planner.plan("open the example site")
        self.assertEqual([a.action for a in result.actions], ["open_app", "open_website"])

    def test_agent_replans_after_failed_step(self):
        first_plan = '''{"actions":[{"action":"open_app","value":"chrome","description":"Open Chrome"}],"meta":{"confidence":0.9}}'''
        second_plan = '''{"actions":[{"action":"open_app","value":"msedge","description":"Open Edge instead"}],"meta":{"confidence":0.9}}'''
        llm = FakeLLM([first_plan, second_plan])
        planner = TaskPlanner(self.cfg, llm)
        executor = FakeExecutor([
            ExecutionReport(False, "failed", "Chrome did not open"),
            ExecutionReport(True, "opened", None, 1),
        ])
        spoken = []
        confirmations = []
        agent = AgentOrchestrator(
            self.cfg,
            planner,
            executor,
            ActionPolicy(self.cfg),
            lambda prompt: confirmations.append(prompt) or "yes",
            spoken.append,
        )
        task = agent.run("open a browser")
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.replans, 1)
        self.assertEqual(executor.calls, ["open_app", "open_app"])

    def test_agent_cancels_sensitive_action(self):
        raw = '''{"actions":[{"action":"send_whatsapp","name":"Aman","message":"hello","description":"Send WhatsApp message"}],"meta":{"confidence":0.9}}'''
        agent = AgentOrchestrator(
            self.cfg,
            TaskPlanner(self.cfg, FakeLLM([raw])),
            FakeExecutor([ExecutionReport(True)]),
            ActionPolicy(self.cfg),
            lambda prompt: "no",
            lambda text: None,
        )
        task = agent.run("message Aman")
        self.assertEqual(task.status, TaskStatus.CANCELLED)

    def test_task_advances_only_on_success(self):
        task = AgentTask(goal="demo")
        task.actions = [ActionRequest(action="open_app", value="chrome")]
        task.advance(StepResult(action="open_app", success=False, error="failed"))
        self.assertEqual(task.current_index, 0)
        task.advance(StepResult(action="open_app", success=True))
        self.assertEqual(task.current_index, 1)


if __name__ == "__main__":
    unittest.main()
