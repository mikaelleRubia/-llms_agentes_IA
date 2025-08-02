from langchain_groq import ChatGroq
from IPython.display import display, Markdown
from dotenv import load_dotenv
load_dotenv()


def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )
  return llm

### Exibição do resultado
def show_res(res):
  display(Markdown(res))
  return res

def format_res(res, return_thinking=False):
  res = res.strip()

  if return_thinking:
    res = res.replace("<think>", "[pensando...] ")
    res = res.replace("</think>", "\n---\n")

  else:
    if "</think>" in res:
      res = res.split("</think>")[-1].strip()

  return res