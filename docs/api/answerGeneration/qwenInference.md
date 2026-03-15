# qwenInference.py

## 概述
`answerGeneration/qwenInference.py` 提供了一种将项目在本地或者内部服务直接连结“Qwen 系列大模型”的基座（在环境根里 `Qwen-model-7B/` 以及利用如 `vLLM` 的加速框架服务或 huggingface transformers 进行本地加载的类库接口协议抽象）。

该组件主要是负责“把组装好的提问与背景资料 Prompt 提交给指定的千问大模型进行推理预测 (Inference)，并按照格式回收最终大语言模型续写输出后的字符串文本”。不同于用 `openai` 库跨外网向云厂商申请：此脚本是特调以兼容其所使用的专用硬件架构或者内部服务端点的专属推理适配器。

## 模块主要方法和类的构成

### `class QwenLocalInference:`
对包含词表，配置文件和 `.safetensors` 参数进行包装挂载的模型封装基底。如果是在内部的显卡配置环境下直接利用 Huggingface `pipeline` 起进程它控制分配（这其中可使用例如 quantization、多卡并跑等逻辑加载策略等）。

#### `__init__(self, model_path, device="cuda")`
载入并构造出带有千问系统词表（`tokenizer`）以及在内存里初始化的带有计算流控制配置结构项的主引擎。它会耗时相对较久以搬运庞大的权重块加载器入 GPU。

#### `_apply_chat_template(messages) -> str | list[int]`
这是为了符合千问针对问答场景 `Chat` （带身份隔离特殊符号如 `<|im_start|>user`, `<|im_start|>assistant` 等指令规范）专门重写的输入文本序列化工具或者对 `tokens` 的模板组装函数（有些库是 `tokenizer.apply_chat_template` ），避免输入进去不包含特殊的对话格式标记导致的胡言乱语。

#### `generate_answer(self, prompt_text: str, max_new_tokens=1024, temperature=0.7) -> str`
同步向底层发起一条计算生成命令请求任务执行者：接收带有系统规则前缀跟长上下文的内容之后，进入本地端模型的大规模线性代数与 softmax 开计算解码节点序列的任务层，一直运行直至命中其指定的生成上限或者到达 `EOS` 截断标记停止。
它能够控制推理层超配参数如随机创造性 `temperature` 或者截断多样性（如 `top_p` 或 `repetition_penalty`）。

### `class QwenAPIClient:`
（如果配置使用的是远端代理点/VLLM 框架启动的服务站时）。作为另一套与外部接口如 OpenAI `v1/chat/completions` 或者千问专属格式对齐的 POST 方法轻量化组件包，通过异步协程或者标准库的 `requests` 实现无显存包袱的网络请求类与配置封装，提供一致的 `generate_answer` 以兼容内部逻辑。

### `get_llm_inference()`
这是一个依据当前系统项目环境配置：`[inference.backend]` 为 “`local`” 或者是 “`api`” 来决定实例化上述两种之一并且全局作为单例对象使用以规避重复导入显存加载缓慢的工厂包裹抽象方法：该设计有效使得对千问大模型在各类不同服务器环境下拥有完全的计算无缝降级或切换调度控制逻辑层（即依赖注入机制容器）。
