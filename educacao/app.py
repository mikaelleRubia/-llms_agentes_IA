import os
import getpass
from utils import load_llm, show_res, format_res
from dotenv import load_dotenv
load_dotenv()


id_model = "deepseek-r1-distill-llama-70b"
temperature = 0.7

# one-shot, foi passado um unico exemplo
prompt = """
Gere 3 perguntas de múltipla escolha sobre o ciclo da água, adequadas para estudantes do ensino fundamental.
Inclua 4 alternativas por pergunta e destaque a resposta correta.
"""


# Few-shot
prompt_01 = """
Aqui estão alguns exemplos de perguntas sobre ciências:

1. Qual é o principal gás responsável pelo efeito estufa?
a) Oxigênio
b) Nitrogênio
c) Dióxido de Carbono
d) Hidrogênio
Resposta: c

Agora, crie 3 novas perguntas de múltipla escolha sobre o ciclo da água.
"""


# Chain-of-Thought
prompt_02 = """
Explique o raciocínio passo a passo antes de responder e detalhe cada fase do processo.
Pergunta: Por que as nuvens se formam no céu?
"""

# exemplos:
#01:  prompt = """
# Resolva o seguinte problema de forma passo a passo:
# Se João tem 3 vezes mais maçãs que Maria, e juntos eles têm 48 maçãs, quantas maçãs cada um tem?
# Pense passo a passo.
# """

#02: prompt = """
# Você é um consultor financeiro. Um cliente tem 100 mil reais para investir. Ele pode escolher entre um fundo de ações com alta volatilidade e um título de renda fixa de baixo risco.
# Pense passo a passo: quais fatores ele deve considerar para tomar a decisão mais adequada ao seu perfil?
# Explique seu raciocínio antes de sugerir uma opção.
# """

#03: prompt = "explique computação quântica para uma criança de 5 anos"

# 04:   prompt = f"""
#   Você é um tutor especialista em {topic}. Gere {quantity} exercícios para um aluno de nível {level}.
# {f"- Apenas caso faça sentido no contexto, adapte de forma natural e sutil os enunciados dos exercícios para refletir a afinidade do aluno com o tema '{interests}'." if interests else ""}
# - Formato dos exercícios: Múltipla escolha com 4 opções.
# - Incluir explicação passo a passo e o raciocínio usado para chegar à resposta.
# - Não use LaTeX e nenhuma sequência iniciada por barra invertida (como \frac, \sqrt, ou similares). Use apenas linguagem natural e símbolos comuns do teclado.

# Exemplo de estrutura:
# 1. [Enunciado]
#    a) Opção 1
#    b) Opção 2
#    c) Opção 3
#    d) Opção 4
#    Resposta: [Letra correta]
#    Explicação: [Passo a passo detalhado]
# """

llm = load_llm(id_model, temperature)

res = llm.invoke(prompt_02)
res_trat = show_res(format_res(res.content, return_thinking=True))

print(res_trat)




