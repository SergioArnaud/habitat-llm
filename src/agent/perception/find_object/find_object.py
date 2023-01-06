from pydantic import BaseModel, Extra
from langchain.chains.base import Chain
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import BaseLLM
from agent.env.environment_interface import EnvironmentInterface


PROMPT = """You're in a house and the following objects are present: 
<object name> is in (<receptacle>) and has id (<object_id>) 
-----------------------------
{objects}
You received a query where you're asked to pick object in a particular receptacle, 
the query might specify an object that is or is not in present in the house, 
the query might also ask for an object in a receptacle that's not actually in:

- If there's an unique object that matches the query provided please return [object_id in <receptacle>].
- If several objects match the query return [object_id in <receptacle>], ... for all the matches.
- If the the object is not on the scene or the receptacle of the query doesn't match
with the receptacle of the object with the objects present in the house please add a 
message explaining so.

Be really carefull to give the exact id that corresponds to the object-receptacle pair!!!

Query: {target_object}
Answer: 
"""


class FindObj(Chain, BaseModel):

    llm: BaseLLM
    env: EnvironmentInterface
    input_key: str = 'target_object'
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
    
    def get_objects_in_environment(self):
        objects = ''
        for receptacle in self.env.scene_parser.grouped_objects.keys():
            for obj in self.env.scene_parser.grouped_objects[receptacle]:
                objects += f'{obj.short_name} is in ({receptacle}) and has id ({obj.name})  \n'
            objects += '-------------------------------  \n'

        return objects

    def from_config(llm, env, verbose=False):
        return FindObj(llm=llm, env=env, verbose=verbose)

    def _call(self, input):
        # Extract objects from environment
        objects = self.get_objects_in_environment()

        # Create prompt
        prompt_template = PROMPT.replace('{objects}', objects)
        prompt = PromptTemplate(template=prompt_template, input_variables=self.input_keys)

        # Execute llm query
        llm_executor = LLMChain(prompt=prompt, llm=self.llm, verbose=self.verbose)
        answer = llm_executor.run(input)
        return {self.output_key: answer}
        