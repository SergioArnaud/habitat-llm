from pydantic import BaseModel, Extra
from langchain.chains.base import Chain
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import BaseLLM
from langchain.input import print_text
from agent.environment_interface import EnvironmentInterface


class ActiveUserFeedback(Chain, BaseModel):
    input_key: str = 'question_to_human'
    output_key: str = 'answer_from_human'

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
    
    def _call(self, question):
        # Extract objects from environment
        answer = input(f'Robot assistant asks: {question}')
        return {self.output_key: answer}
    
    def from_config():
        return ActiveUserFeedback()
        