from pprint import pprint
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import Tool

# Local
from agent.env.environment_interface import EnvironmentInterface   # noqa: F401
from agent.feedback.active_user_feedback import ActiveUserFeedback  # noqa: F401
from agent.env.skill_executor import SkillExecutor  # noqa: F401
from agent.perception import FindObj  # noqa: F261


class HighLevelLLM:
    # Add this parameters to conf
    def __init__(
        self,
        conf,
        env,
        temperature=0,
        model_name="text-davinci-003",
        max_tokens=400,
        verbose=True,
    ):
        self.conf = conf
        self.agent_conf = conf.agent
        self.device = env.device
        self.env = env
        self.verbose = verbose

        # Initialize llm (add to conf)
        self.llm = OpenAI(temperature=0, model_name=model_name, max_tokens=max_tokens)

        # Initialize modules
        self.tools = []
        self.initialize_motor_skills()
        self.initialize_perception_modules()
        self.initialize_feedback_modules()

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=self.verbose,
        )

    # TODO(S): Unify initialization
    def initialize_motor_skills(self):
        for skill_name in self.agent_conf.motor_skills:
            conf = self.agent_conf.motor_skills[skill_name]
            if conf.agent_tool:
                skill = SkillExecutor(self.env, self.device, skill_name)
                self.tools.append(
                    Tool(name=conf.name, description=conf.description, func=skill.run)
                )

    def initialize_perception_modules(self):
        for perception_module in self.agent_conf.perception:
            conf = self.agent_conf.perception[perception_module]
            if conf.agent_tool:
                cls = eval(conf.module)
                module = cls.from_config(
                    llm=self.llm, env=self.env, verbose=conf.verbose
                )
                self.tools.append(
                    Tool(name=conf.name, description=conf.description, func=module.run)
                )

    def initialize_feedback_modules(self):
        for feedback_module in self.agent_conf.feedback:
            conf = self.agent_conf.feedback[feedback_module]
            if conf.agent_tool:
                cls = eval(conf.module)
                module = cls.from_config()
                self.tools.append(
                    Tool(name=conf.name, description=conf.description, func=module.run)
                )

    def run(self, query):
        return self.agent.run(query)

    def show_scene(self):
        """
        For now this is just printing the objects and receptacles
        """
        pprint(self.env.scene_parser.grouped_objects)
