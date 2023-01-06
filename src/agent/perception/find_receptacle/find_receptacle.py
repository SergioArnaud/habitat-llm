from pydantic import BaseModel, Extra
from langchain.chains.base import Chain
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import BaseLLM
from agent.env.environment_interface import EnvironmentInterface


PROMPT = """You're in a house and the following objects are present: 
<receptacle> with id [receptacle_id]
-----------------------------
{receptacles}

You received a query where you're asked to go to a particular receptacle, 
the query might specify a receptacle that is or is not in present in the house.

- If there's an unique receptacle that matches the query provided please return it's id.
- If several objects match the query return all their ids separated by a comma.
- If the the receptacle is not on the scene please answer with a message explaining so.

Query: {target_receptacle}
Answer: 
"""


class FindReceptacle(Chain, BaseModel):

    llm: BaseLLM
    env: EnvironmentInterface
    input_key: str = 'target_receptacle'
    output_key: str = 'answer'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self):
        """Expect input key.
        """
        return [self.input_key]

    @property
    def output_keys(self):
        """Expect output key.
        """
        return [self.output_key]
    
    def get_receptacles_in_environment(self):
        receptacles = ''
        for receptacle_name in self.env.scene_parser.grouped_receptacles.keys():
            for receptacle in self.env.scene_parser.grouped_receptacles[receptacle_name]:
                receptacles += f'{receptacle_name} with it [{receptacle.name}]\n'
        return receptacles

    def _call(self, input):
        # Extract objects from environment
        objects = self.get_receptacles_in_environment()

        # Create prompt
        prompt_template = PROMPT.replace('{receptacles}', objects)
        prompt = PromptTemplate(template=prompt_template, input_variables=self.input_keys)

        # Execute llm query
        llm_executor = LLMChain(prompt=prompt, llm=self.llm, verbose=False)
        answer = llm_executor.run(input)
        return {self.output_key: answer}
        