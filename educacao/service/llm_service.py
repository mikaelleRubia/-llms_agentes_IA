from langchain_core.prompts import PromptTemplate
from IPython.display import display, Markdown
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LlmService:
    def __init__(self, model_name="deepseek-r1-distill-llama-70b",temperature=1):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_template = PromptTemplate.from_template(
            """
            {input}
            ---
            Contexto: {context}
            """
        )

    def run_prompt(self, prompt: str, context: str) -> str:
        chain = self.prompt_template | self.llm
        response = chain.invoke({"input": prompt, "context": context})

        return response.content if hasattr(response, "content") else response
    

class FormatResponse:
    @staticmethod
    def show_res(res):
        display(Markdown(res))
        return res
    
    @staticmethod
    def format_res(res, return_thinking=False):
        res = res.strip()

        if return_thinking:
            res = res.replace("<think>", "[pensando...] ")
            res = res.replace("</think>", "\n---\n")

        else:
            if "</think>" in res:
                res = res.split("</think>")[-1].strip()

            return res 