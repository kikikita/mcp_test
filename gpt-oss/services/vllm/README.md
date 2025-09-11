# Так получилось, что необходимо было изменить вручную в .venv библиотеку vllm

---
vllm==0.10.1

---
# Пути где менять
```.venv/lib/python3.10/site-packages/vllm/entrypoints/harmony_utils.py```

```.venv/lib/python3.10/site-packages/vllm/entrypoints/openai/serving_chat.py```

```.venv/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/__init__.py```

```.venv/lib/python3.10/site-packages/vllm/entrypoints/openai/tool_parsers/openai_tool_parser.py```

```.venv/lib/python3.10/site-packages/vllm/reasoning/gptoss_reasoning_parser.py```

---
## Команда запуска модели
```vllm serve openai/gpt-oss-120b --tensor-parallel-size 2 --tool-call-parser openai --reasoning-parser openai_gptoss --enable-auto-tool-choice```
