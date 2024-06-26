# autologic
Implementación en python del framework propuesto en el paper SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures. Fork de waszumteufel/autologic, errores corregidos y adaptado para funcionar con Azure OpenAI 

## Instalación
```bash
pip install -r requirements.txt
```

## Framework Overview

The SELF-DISCOVER framework consists of two key stages:

Stage 1: Self-discover a reasoning structure for the task from a set of seed "reasoning modules"

Stage 2: Solve instances by following the composed structure

The first stage has 3 steps guided by meta-prompting:

1. SELECT relevant reasoning modules
2. ADAPT modules to be task-specific
3. IMPLEMENT structure with adapted modules

## OpenAI API Key
Crear un archivo .env en la raíz del proyecto con la siguiente estructura:
```bash
MODEL_GPT3=...
OPENAI_KEY_3=...
ENDPOINT_GPT3=...

MODEL_GPT3_16K=...
OPENAI_KEY_3_16K=...
ENDPOINT_GPT3_16K=...

MODEL_GPT4=...
OPENAI_KEY_4=...
ENDPOINT_GPT4=...
```

## Ejemplo de uso
```python
from autologic import reasoningEngine

llmConfig = reasoningEngine.LLMConfig(
    model_type = reasoningEngine.ModelType.OPENAI,
    model_name = "GPT4", # You can use any model name available to you through openai - omitting the model_name will default to gpt-3.5-turbo-0125
    temp = 0.2,
    context_length = 2000
)

problem_task = "Beth and Sam are 500 miles apart. If Beth travels at 60mph and leaves her house at 1pm, what time will she arrive at Sam's house?" # 9:20PM

answer = reasoningEngine.solve(task = problem_task,verbose=False,discover_config=llmConfig, print_in_out=True)
print(answer)
```