
from dotenv import load_dotenv
load_dotenv()



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