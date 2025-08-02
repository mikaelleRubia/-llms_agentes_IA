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

# Função para criar o prompt
def build_prompt(topic, quantity, level, interests):
    prompt = f"""
Você é um tutor especialista em {topic}. Gere {quantity} exercícios para um aluno de nível {level}.
{f"- Apenas caso faça sentido no contexto, adapte de forma natural e sutil os enunciados dos exercícios para refletir a afinidade do aluno com o tema '{interests}'." if interests else ""}
- Formato dos exercícios: Múltipla escolha com 4 opções.
- Incluir explicação passo a passo e o raciocínio usado para chegar à resposta.
- Não use LaTeX e nenhuma sequência iniciada por barra invertida (como \frac, \sqrt, ou similares). Use apenas linguagem natural e símbolos comuns do teclado.

Exemplo de estrutura:
1. [Enunciado]
   a) Opção 1
   b) Opção 2
   c) Opção 3
   d) Opção 4
   Resposta: [Letra correta]
   Explicação: [Passo a passo detalhado de como chegou ao resultado]
"""
    return prompt