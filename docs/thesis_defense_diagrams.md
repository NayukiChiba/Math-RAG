# Math-RAG 开题答辩图表集

> 以下所有图表均基于项目源代码的真实架构，可直接用于开题答辩 PPT。

---

## 图 1：系统总体架构图

```mermaid
graph TB
    subgraph 输入层["📥 输入层"]
        PDF["📄 数学教材 PDF"]
        UserQ["👤 用户提问"]
    end

    subgraph 数据处理层["⚙️ 数据处理层"]
        OCR["Pix2Text OCR<br/>PDF → Markdown"]
        TermExtract["术语抽取<br/>DeepSeek API"]
        TermClean["术语清洗与过滤"]
        DataGen["结构化数据生成<br/>JSON Schema"]
    end

    subgraph 索引层["📦 索引层"]
        BM25Idx["BM25/BM25+ 索引<br/>rank-bm25"]
        VecIdx["FAISS 向量索引<br/>sentence-transformers"]
        Corpus["语料库 corpus.jsonl"]
    end

    subgraph 检索层["🔍 检索层"]
        QRewrite["查询改写<br/>同义词扩展"]
        BM25Search["BM25+ 检索"]
        VecSearch["向量检索<br/>BGE-base-zh"]
        DirectLookup["直接术语查找"]
        Fusion["分数融合<br/>RRF / 加权"]
    end

    subgraph 生成层["🤖 生成层"]
        ScopeCheck["领域判断 &<br/>拒答策略"]
        PromptBuild["Prompt 构建<br/>System + Context + Query"]
        QwenLLM["Qwen2.5-Math-7B<br/>本地推理"]
    end

    subgraph 输出层["📤 输出层"]
        Answer["📝 结构化回答<br/>术语 + 答案 + 来源"]
        WebUI["🌐 Gradio WebUI"]
        Report["📊 评测报告"]
    end

    PDF --> OCR --> TermExtract --> TermClean --> DataGen
    DataGen --> Corpus
    Corpus --> BM25Idx
    Corpus --> VecIdx

    UserQ --> QRewrite
    QRewrite --> BM25Search
    QRewrite --> VecSearch
    QRewrite --> DirectLookup
    BM25Idx -.-> BM25Search
    VecIdx -.-> VecSearch
    BM25Search --> Fusion
    VecSearch --> Fusion
    DirectLookup --> Fusion

    Fusion --> ScopeCheck
    ScopeCheck -->|"域内"| PromptBuild
    ScopeCheck -->|"域外"| Answer
    PromptBuild --> QwenLLM --> Answer
    Answer --> WebUI
    Answer --> Report
```

---

## 图 2：数据处理流水线

```mermaid
flowchart LR
    subgraph Stage1["阶段一：OCR"]
        A1["PDF 教材"] -->|"PyMuPDF 渲染"| A2["逐页图像"]
        A2 -->|"Pix2Text"| A3["Markdown 文本<br/>含 LaTeX 公式"]
    end

    subgraph Stage2["阶段二：术语抽取"]
        A3 -->|"文本分块"| B1["文本块<br/>≤1800 字符"]
        B1 -->|"DeepSeek API"| B2["候选术语<br/>JSON 数组"]
        B2 -->|"清洗/过滤"| B3["术语列表 all.json<br/>页码映射 map.json"]
    end

    subgraph Stage3["阶段三：结构化生成"]
        B3 -->|"OCR 上下文"| C1["Prompt 构建<br/>含示例 JSON"]
        C1 -->|"DeepSeek API"| C2["术语 JSON<br/>含定义/公式/用法"]
        C2 -->|"质量检查"| C3{"合格?"}
        C3 -->|"是"| C4["保存 term.json"]
        C3 -->|"否"| C5["修复 Prompt"] --> C1
    end

    subgraph Stage4["阶段四：索引构建"]
        C4 --> D1["语料构建<br/>corpus.jsonl"]
        D1 --> D2["BM25+ 索引<br/>混合分词 + N-gram"]
        D1 --> D3["FAISS 向量索引<br/>BGE-base-zh-v1.5"]
    end

    style Stage1 fill:#E3F2FD,stroke:#1565C0
    style Stage2 fill:#E8F5E9,stroke:#2E7D32
    style Stage3 fill:#FFF3E0,stroke:#E65100
    style Stage4 fill:#F3E5F5,stroke:#6A1B9A
```

---

## 图 3：混合检索策略架构

```mermaid
flowchart TB
    Q["用户查询"] --> QR["QueryRewriter<br/>同义词扩展"]
    QR --> Q1["原始查询"]
    QR --> Q2["扩展查询 1"]
    QR --> Q3["扩展查询 N"]

    Q1 --> BM25["BM25+ 检索器"]
    Q2 --> BM25
    Q3 --> BM25
    Q1 --> Vec["向量检索器<br/>FAISS + BGE"]
    Q2 --> Vec
    Q3 --> Vec
    Q1 --> DL["直接术语查找<br/>精确匹配"]

    BM25 --> |"Top-K × recall_factor"| Norm1["百分位数归一化"]
    Vec --> |"Top-K × recall_factor"| Norm2["百分位数归一化"]

    Norm1 --> FuseW["加权融合<br/>α·BM25 + β·Vector"]
    Norm2 --> FuseW
    Norm1 --> FuseR["RRF 融合<br/>1/(k + rank)"]
    Norm2 --> FuseR

    FuseW --> Merge["结果合并<br/>直接查找优先"]
    FuseR --> Merge
    DL --> Merge

    Merge --> TopK["返回 Top-K 结果"]

    style Q fill:#BBDEFB,stroke:#1565C0
    style FuseW fill:#C8E6C9,stroke:#2E7D32
    style FuseR fill:#C8E6C9,stroke:#2E7D32
    style DL fill:#FFE0B2,stroke:#E65100
```

---

## 图 4：RAG 问答流程（核心 Pipeline）

```mermaid
sequenceDiagram
    participant U as 👤 用户
    participant P as RagPipeline
    participant D as 领域判断
    participant R as 混合检索器
    participant E as 证据过滤
    participant T as Prompt 模板
    participant Q as Qwen2.5-Math

    U->>P: query("什么是一致收敛？")
    P->>D: isMathDomainQuery()
    D-->>P: ✅ 数学领域

    P->>R: retrieve(query, topK=5)
    R->>R: BM25+ 检索
    R->>R: 向量检索
    R->>R: 直接术语查找
    R->>R: 加权融合
    R-->>P: 检索结果列表

    P->>E: shouldRefuseOutOfScope()
    E->>E: 计算 Top Score
    E->>E: 词汇重叠检查
    E-->>P: ✅ 域内可回答

    P->>P: enrichResults() 补充完整文本
    P->>T: buildMessages(query, results)
    T->>T: formatTermContext() × N
    T->>T: 拼接上下文 ≤ 4000 字符
    T-->>P: [system, user] messages

    P->>Q: generateFromMessages()
    Q->>Q: apply_chat_template
    Q->>Q: model.generate()
    Q-->>P: 生成的回答

    P-->>U: {query, terms, answer, sources, latency}
```

---

## 图 5：项目模块依赖关系图

```mermaid
graph LR
    subgraph 核心模块["核心模块"]
        Config["config.py<br/>config.toml"]
        Utils["utils/<br/>fileLoader<br/>outputManager"]
    end

    subgraph 数据模块["数据处理模块"]
        DG["dataGen/<br/>pix2text_ocr<br/>extract_terms<br/>filter_terms<br/>data_gen"]
        DS["dataStat/<br/>统计 & 可视化"]
    end

    subgraph 检索模块["检索模块"]
        CB["corpusBuilder/<br/>语料构建"]
        QR["queryRewriter/<br/>查询改写"]
        RM["retrieverModules/<br/>BM25 | Vector<br/>Hybrid | HybridPlus"]
    end

    subgraph 生成模块["生成模块"]
        PT["promptTemplates<br/>提示词模板"]
        QI["qwenInference<br/>模型推理"]
        RP["ragPipeline<br/>端到端 RAG"]
        WUI["webui<br/>Gradio 界面"]
    end

    subgraph 评测模块["评测模块"]
        EQ["evaluationData/<br/>查询生成"]
        ME["modelEvaluation/<br/>quickEval<br/>retrievalEval<br/>generationEval"]
    end

    Config --> DG
    Config --> RM
    Config --> PT
    Config --> QI
    Utils --> DG
    Utils --> CB
    Utils --> RP

    DG --> CB
    CB --> RM
    QR --> RM
    RM --> RP
    PT --> RP
    QI --> RP
    RP --> WUI
    RP --> ME
    EQ --> ME
    DS --> ME

    style Config fill:#FFF9C4,stroke:#F57F17
    style RP fill:#E1BEE7,stroke:#6A1B9A
```

---

## 图 6：检索器类继承与组合关系

```mermaid
classDiagram
    class BM25Retriever {
        +corpusFile: str
        +indexFile: str
        +buildIndex()
        +loadIndex()
        +search(query, topK)
    }

    class BM25PlusRetriever {
        +termsMap: dict
        +queryRewriter: QueryRewriter
        +buildIndex()
        +search(query, topK, expandQuery)
        +directLookup(terms, baseScore)
        +getExpandedTerms(query)
    }

    class VectorRetriever {
        +modelName: str
        +faissIndex: FAISS
        +encode(texts)
        +buildIndex()
        +search(query, topK)
    }

    class HybridRetriever {
        +bm25: BM25Retriever
        +vector: VectorRetriever
        +fuseRRF()
        +fuseWeighted()
        +search(query, topK, strategy)
    }

    class HybridPlusRetriever {
        +bm25: BM25PlusRetriever
        +vector: VectorRetriever
        +normalizePercentile()
        +fuseRRFImproved()
        +fuseWeightedImproved()
        +search(query, topK, strategy)
    }

    class QueryRewriter {
        +termsMap: dict
        +rewrite(query, maxTerms)
        +rewriteBatch(queries)
    }

    class RerankerRetriever {
        +rerankerModel: str
        +rerank(query, candidates)
    }

    BM25Retriever <|-- BM25PlusRetriever
    BM25Retriever --o HybridRetriever
    VectorRetriever --o HybridRetriever
    BM25PlusRetriever --o HybridPlusRetriever
    VectorRetriever --o HybridPlusRetriever
    QueryRewriter --o BM25PlusRetriever
    HybridPlusRetriever --> RerankerRetriever : 可选重排
```

---

## 图 7：分数融合机制详解

```mermaid
flowchart TB
    subgraph 输入["检索结果输入"]
        BM25R["BM25+ 结果<br/>Top-50"]
        VecR["向量结果<br/>Top-50"]
    end

    subgraph 归一化["分数归一化"]
        direction TB
        P["百分位数归一化<br/>rank/n → [0,1]"]
        MM["Min-Max 归一化<br/>(s-min)/(max-min)"]
        ZS["Z-Score 归一化<br/>(s-μ)/σ"]
    end

    subgraph 自适应权重["自适应权重计算"]
        OL["计算结果重叠度<br/>overlap_ratio"]
        OL -->|"> 0.5"| EQ["均等权重<br/>α=0.5, β=0.5"]
        OL -->|"≤ 0.5"| AD["按分数均值分配<br/>α=avg_bm25/total<br/>β=avg_vec/total"]
    end

    subgraph 融合策略["融合策略"]
        WF["加权融合<br/>score = α·s_bm25 + β·s_vec"]
        RF["RRF 融合<br/>score = Σ 1/(k+rank)<br/>k 动态调整"]
    end

    BM25R --> P
    VecR --> P
    BM25R --> MM
    VecR --> MM
    BM25R --> ZS
    VecR --> ZS

    P --> OL
    MM --> OL
    ZS --> OL

    EQ --> WF
    AD --> WF
    EQ --> RF
    AD --> RF

    WF --> Result["排序后 Top-K"]
    RF --> Result

    style 归一化 fill:#E3F2FD,stroke:#1565C0
    style 自适应权重 fill:#E8F5E9,stroke:#2E7D32
    style 融合策略 fill:#FFF3E0,stroke:#E65100
```

---

## 图 8：域外拒答决策流程

```mermaid
flowchart TB
    Start["用户查询"] --> MathCheck{"包含数学<br/>关键词/符号？"}
    MathCheck -->|"否"| Refuse1["🚫 直接拒答<br/>我不知道。"]
    MathCheck -->|"是"| Retrieve["执行混合检索"]

    Retrieve --> Empty{"检索结果<br/>为空？"}
    Empty -->|"是"| Refuse2["🚫 拒答<br/>无证据"]
    Empty -->|"否"| TopScore["计算 Top Score"]

    TopScore --> Overlap{"查询与术语<br/>有词汇重叠？"}

    Overlap -->|"有重叠"| RelaxThresh{"Top Score<br/>≥ 0.45？"}
    RelaxThresh -->|"否"| Refuse3["🚫 拒答<br/>证据不足"]
    RelaxThresh -->|"是"| Accept["✅ 生成回答"]

    Overlap -->|"无重叠"| StrictThresh{"Top Score<br/>≥ 0.88？"}
    StrictThresh -->|"否"| Refuse4["🚫 拒答<br/>无匹配证据"]
    StrictThresh -->|"是"| Accept

    style Refuse1 fill:#FFCDD2,stroke:#C62828
    style Refuse2 fill:#FFCDD2,stroke:#C62828
    style Refuse3 fill:#FFCDD2,stroke:#C62828
    style Refuse4 fill:#FFCDD2,stroke:#C62828
    style Accept fill:#C8E6C9,stroke:#2E7D32
```

---

## 图 9：Prompt 模板构建流程

```mermaid
flowchart LR
    subgraph 检索结果
        R1["rank=1: 一致收敛<br/>数学分析 p.89"]
        R2["rank=2: 逐点收敛<br/>数学分析 p.85"]
        R3["rank=3: 函数列<br/>数学分析 p.83"]
    end

    subgraph 格式化["formatTermContext"]
        R1 --> F1["「数学分析」一致收敛<br/>【数学分析(第5版)上 p.89】<br/>定义：函数列 fn 在 D 上..."]
        R2 --> F2["「数学分析」逐点收敛<br/>【数学分析(第5版)上 p.85】<br/>定义：对于固定 x..."]
        R3 --> F3["「数学分析」函数列<br/>【数学分析(第5版)上 p.83】<br/>定义：设 f1, f2, ..."]
    end

    subgraph 拼接["buildContext"]
        F1 --> CTX["上下文字符串<br/>≤ 4000 字符<br/>分隔符: ---"]
        F2 --> CTX
        F3 --> CTX
    end

    subgraph 组装["buildMessages"]
        SP["System Prompt<br/>数学教学助手角色<br/>6 条回答规则"]
        CTX --> UP["User Prompt<br/>参考资料 + 问题"]
        SP --> MSG["[system, user]<br/>messages 列表"]
        UP --> MSG
    end

    MSG --> LLM["Qwen2.5-Math 推理"]
```

---

## 图 10：术语数据 JSON Schema

```mermaid
erDiagram
    TERM_RECORD {
        string id "ma-柯西列"
        string term "柯西列"
        array aliases "Cauchy 列"
        string sense_id "1"
        string subject "数学分析"
        string notation "LaTeX 符号"
        array formula "LaTeX 公式列表"
        string usage "用法说明"
        string applications "应用场景"
        string disambiguation "消歧说明"
        array related_terms "关联术语 ≥3"
        array sources "来源页码"
        array search_keys "搜索关键词"
        string lang "zh"
        string confidence "high/medium/low"
    }

    DEFINITION {
        string type "strict/alternative/informal"
        string text "定义文本含 LaTeX"
        string conditions "前提条件"
        string notation "符号说明"
        string reference "来源引用"
    }

    CORPUS_ITEM {
        string doc_id "文档唯一标识"
        string term "术语名称"
        string subject "学科分类"
        string text "拼接后的完整文本"
        string source "教材名称"
        int page "首次出现页码"
    }

    TERM_RECORD ||--o{ DEFINITION : "definitions ≥ 2"
    TERM_RECORD ||--|| CORPUS_ITEM : "构建语料"
```

---

## 图 11：评测体系架构

```mermaid
flowchart TB
    subgraph 数据准备["评测数据准备"]
        QGen["查询生成<br/>generateQueries"]
        TMap["术语映射构建<br/>buildTermMapping"]
        Gold["Golden 标准集<br/>queries.jsonl"]
        QGen --> Gold
        TMap --> Gold
    end

    subgraph 检索评测["检索质量评测"]
        Quick["快速评测 quickEval<br/>小数据集"]
        Full["正式评测 evalRetrieval<br/>大规模"]
        Quick --> RM1["Recall@K"]
        Quick --> RM2["MRR"]
        Full --> RM1
        Full --> RM2
        Full --> RM3["NDCG"]
        Full --> RM4["Precision@K"]
    end

    subgraph 生成评测["生成质量评测"]
        GenEval["evalGeneration"]
        GenEval --> GM1["ROUGE-L"]
        GenEval --> GM2["裁判模型评分"]
        GenEval --> GM3["准确性"]
        GenEval --> GM4["完整性"]
    end

    subgraph 对比实验["端到端实验"]
        Exp["Experiments<br/>配置矩阵"]
        Exp --> E1["BM25 vs Vector vs Hybrid"]
        Exp --> E2["不同 α β 权重"]
        Exp --> E3["不同 Top-K"]
        Exp --> E4["查询改写开/关"]
    end

    subgraph 报告["结果汇总"]
        Rep["generateReport"]
        Sig["significanceTest<br/>统计显著性"]
        RM1 --> Rep
        RM2 --> Rep
        GM1 --> Rep
        E1 --> Rep
        Rep --> Sig
        Rep --> OUT["📊 Markdown 报告<br/>+ 图表"]
    end

    Gold --> Quick
    Gold --> Full
    Gold --> GenEval

    style 数据准备 fill:#E3F2FD,stroke:#1565C0
    style 检索评测 fill:#E8F5E9,stroke:#2E7D32
    style 生成评测 fill:#FFF3E0,stroke:#E65100
    style 对比实验 fill:#F3E5F5,stroke:#6A1B9A
    style 报告 fill:#FFF9C4,stroke:#F57F17
```

---

## 图 12：技术栈与工具链全景图

```mermaid
mindmap
  root((Math-RAG))
    数据处理
      Pix2Text OCR
        PyMuPDF
        PIL
      DeepSeek API
        术语抽取
        结构化生成
      数据清洗
        正则过滤
        噪声词库
    检索引擎
      BM25/BM25+
        rank-bm25
        混合分词
        字符 N-gram
      向量检索
        sentence-transformers
        FAISS
        BGE-base-zh-v1.5
      融合策略
        RRF
        加权融合
        百分位数归一化
    生成模型
      Qwen2.5-Math-7B
        Transformers
        PyTorch
        AWQ 量化
      Prompt 工程
        System + Context
        Jinja2 模板
    评测框架
      检索指标
        Recall
        MRR
        NDCG
      生成指标
        ROUGE-L
        裁判模型
      统计检验
        显著性测试
    工程化
      配置管理
        TOML
        Python 配置层
      CLI 入口
        统一命令行
        子命令路由
      Web 界面
        Gradio
```

---

## 图 13：语料构建与索引流程

```mermaid
flowchart TB
    subgraph 输入["术语 JSON 文件"]
        TJ1["柯西列.json"]
        TJ2["一致收敛.json"]
        TJ3["极限.json"]
        TJN["..."]
    end

    subgraph 语料构建["corpusBuilder"]
        LJ["loadJsonFile<br/>加载术语 JSON"]
        BT["buildTextFromTerm<br/>拼接文本"]
        ECI["extractCorpusItem<br/>提取语料条目"]
        BC["buildCorpus<br/>批量构建"]
        VAL["validateCorpusFile<br/>验证完整性"]
    end

    subgraph 文本拼接["文本拼接规则"]
        direction TB
        T1["术语: 一致收敛"]
        T2["别名: uniform convergence"]
        T3["定义1「strict」: ..."]
        T4["定义2「alternative」: ..."]
        T5["公式: \\forall ε > 0 ..."]
        T6["用法: ..."]
        T7["消歧: ..."]
    end

    subgraph 索引["索引构建"]
        CJ["corpus.jsonl"]
        BM25B["BM25+ 索引构建<br/>混合分词"]
        FAISSB["FAISS 索引构建<br/>BGE 编码"]
        BM25F["bm25_index.pkl<br/>bm25plus_index.pkl"]
        FAISSF["vector_index.faiss<br/>vector_embeddings.npz"]
    end

    TJ1 --> LJ
    TJ2 --> LJ
    TJ3 --> LJ
    TJN --> LJ

    LJ --> BT --> ECI --> BC --> VAL
    BT -.-> T1
    BT -.-> T2
    BT -.-> T3
    BT -.-> T4
    BT -.-> T5
    BT -.-> T6
    BT -.-> T7

    VAL --> CJ
    CJ --> BM25B --> BM25F
    CJ --> FAISSB --> FAISSF

    style 文本拼接 fill:#FFF9C4,stroke:#F57F17
    style 索引 fill:#E8F5E9,stroke:#2E7D32
```

---

## 图 14：CLI 命令路由架构

```mermaid
flowchart TB
    CLI["math-rag / python mathRag.py"]

    CLI --> ingest["ingest<br/>PDF 全流程入库"]
    CLI --> buildIdx["build-index<br/>索引重建"]
    CLI --> rag["rag<br/>RAG 问答"]
    CLI --> genQ["generate-queries<br/>评测查询生成"]
    CLI --> btm["build-term-mapping<br/>术语映射构建"]
    CLI --> qe["quick-eval<br/>快速检索评测"]
    CLI --> er["eval-retrieval<br/>正式检索评测"]
    CLI --> exp["experiments<br/>端到端对比实验"]
    CLI --> eg["eval-generation<br/>生成质量评测"]
    CLI --> rep["report<br/>报告生成"]
    CLI --> serve["serve<br/>WebUI 启动"]
    CLI --> stats["stats<br/>数据统计"]

    ingest --> |"OCR"| P1["pix2text_ocr"]
    ingest --> |"抽取"| P2["extract_terms"]
    ingest --> |"生成"| P3["data_gen"]
    ingest --> |"索引"| P4["buildCorpus"]

    rag --> |"单条"| R1["pipeline.query()"]
    rag --> |"批量"| R2["pipeline.batchQuery()"]

    serve --> |"webui"| S1["Gradio App"]
    serve --> |"experiment"| S2["实验回放界面"]

    style CLI fill:#BBDEFB,stroke:#1565C0
    style ingest fill:#C8E6C9,stroke:#2E7D32
    style rag fill:#E1BEE7,stroke:#6A1B9A
```

---

## 图 15：BM25+ 改进检索器详解

```mermaid
flowchart TB
    subgraph 传统BM25["传统 BM25"]
        T1["单一分词方式"]
        T2["固定查询"]
        T3["无术语映射"]
    end

    subgraph BM25Plus["BM25+ 改进"]
        direction TB
        I1["混合分词策略<br/>词级 + 字符级"]
        I2["字符 N-gram<br/>2-gram + 3-gram"]
        I3["查询改写<br/>同义词扩展"]
        I4["直接术语查找<br/>精确匹配"]
        I5["评测感知术语映射<br/>termsMap"]
    end

    subgraph 效果["改进效果"]
        E1["✅ 解决中文分词歧义"]
        E2["✅ 提升短查询召回率"]
        E3["✅ 覆盖术语别名"]
        E4["✅ 精确命中已知术语"]
    end

    T1 -.->|"改进"| I1
    T1 -.->|"改进"| I2
    T2 -.->|"改进"| I3
    T3 -.->|"改进"| I4
    T3 -.->|"改进"| I5

    I1 --> E1
    I2 --> E2
    I3 --> E3
    I4 --> E4
    I5 --> E4

    style 传统BM25 fill:#FFCDD2,stroke:#C62828
    style BM25Plus fill:#C8E6C9,stroke:#2E7D32
    style 效果 fill:#E3F2FD,stroke:#1565C0
```

---

## 图 16：数据目录结构总览

```mermaid
graph TB
    Root["Math-RAG/"]

    Root --> data["data/"]
    data --> raw["raw/<br/>📄 原始 PDF 教材"]
    data --> processed["processed/"]
    data --> evaluation["evaluation/<br/>📋 评测数据"]
    data --> stats_d["stats/<br/>📊 统计结果"]

    processed --> ocr["ocr/{书名}/pages/<br/>📝 逐页 Markdown"]
    processed --> terms["terms/{书名}/<br/>📑 all.json + map.json"]
    processed --> chunk["chunk/{书名}/<br/>📦 术语 JSON 文件"]
    processed --> retrieval["retrieval/<br/>corpus.jsonl<br/>bm25_index.pkl<br/>vector_index.faiss"]

    evaluation --> queries["queries.jsonl"]
    evaluation --> term_map["term_mapping.json"]
    evaluation --> golden["golden set"]

    Root --> outputs["outputs/"]
    outputs --> rag_res["rag_results.jsonl"]
    outputs --> reports["reports/<br/>📊 评测报告 + 图表"]
    outputs --> logs["log/<br/>📝 运行日志"]
    outputs --> figures["figures/<br/>📈 可视化图表"]

    style raw fill:#FFECB3,stroke:#FF8F00
    style retrieval fill:#C8E6C9,stroke:#2E7D32
    style reports fill:#E1BEE7,stroke:#6A1B9A
```

---

## 图 17：Qwen 模型推理流程

```mermaid
flowchart TB
    Init["QwenInference 初始化"]
    Init --> LoadCfg["读取 config.toml<br/>generation 配置"]
    LoadCfg --> LazyLoad["延迟加载模型"]

    LazyLoad --> CheckQuant{"模型是否<br/>AWQ 量化？"}
    CheckQuant -->|"是"| CUDA["强制 CUDA:0<br/>整模型放单卡"]
    CheckQuant -->|"否"| AutoMap["device_map='auto'<br/>自动分配"]
    CheckQuant -->|"无 GPU"| CPU["CPU 推理<br/>float32"]

    CUDA --> LoadModel["AutoModelForCausalLM<br/>.from_pretrained()"]
    AutoMap --> LoadModel
    CPU --> LoadModel

    LoadModel --> Ready["模型就绪"]

    Ready --> GenMsg["generateFromMessages()"]
    GenMsg --> Template["apply_chat_template<br/>对话格式化"]
    Template --> Tokenize["tokenizer 编码"]
    Tokenize --> Generate["model.generate()<br/>temperature, top_p<br/>max_new_tokens"]
    Generate --> Decode["tokenizer.decode<br/>去除输入，提取生成部分"]
    Decode --> Output["返回回答文本"]

    style Init fill:#E3F2FD,stroke:#1565C0
    style Ready fill:#C8E6C9,stroke:#2E7D32
    style Output fill:#E1BEE7,stroke:#6A1B9A
```

---

## 图 18：实验对比维度矩阵

```mermaid
quadrantChart
    title 检索策略对比维度
    x-axis "召回率 (Recall)" --> "高"
    y-axis "精确度 (Precision)" --> "高"
    quadrant-1 理想区域
    quadrant-2 高精度低召回
    quadrant-3 待改进区域
    quadrant-4 高召回低精度
    "BM25": [0.45, 0.6]
    "Vector": [0.55, 0.5]
    "Hybrid-RRF": [0.7, 0.65]
    "HybridPlus": [0.8, 0.75]
```

> ⚠️ 上图数据为示意，请替换为你的实验真实数据。

---

## 图 19：查询改写同义词扩展示意

```mermaid
flowchart LR
    subgraph 原始查询
        Q["什么是一致收敛？"]
    end

    subgraph 同义词典["MATH_SYNONYMS 同义词典"]
        S1["一致收敛 →<br/>均匀收敛<br/>uniform convergence"]
        S2["极限 →<br/>limit<br/>趋近"]
        S3["导数 →<br/>微商<br/>derivative"]
    end

    subgraph 扩展结果["扩展后查询集"]
        E1["一致收敛"]
        E2["均匀收敛"]
        E3["uniform convergence"]
    end

    Q --> S1
    S1 --> E1
    S1 --> E2
    S1 --> E3

    E1 --> BM25["BM25+ 多路检索"]
    E2 --> BM25
    E3 --> BM25

    style 同义词典 fill:#FFF9C4,stroke:#F57F17
    style 扩展结果 fill:#C8E6C9,stroke:#2E7D32
```

---

## 图 20：系统部署架构

```mermaid
graph TB
    subgraph 用户端["用户端"]
        Browser["🌐 浏览器"]
    end

    subgraph 服务端["服务端（本地 GPU 机器）"]
        Gradio["Gradio Server<br/>Port 7860"]
        Pipeline["RagPipeline"]

        subgraph 模型["模型层"]
            Qwen["Qwen2.5-Math-7B<br/>AWQ 量化"]
            BGE["BGE-base-zh-v1.5<br/>Sentence Transformer"]
        end

        subgraph 存储["存储层"]
            FAISS["FAISS 向量索引"]
            BM25P["BM25+ 倒排索引"]
            Corpus2["语料文件<br/>corpus.jsonl"]
        end
    end

    subgraph 外部服务["外部服务"]
        DSApi["DeepSeek API<br/>数据生成阶段"]
    end

    Browser <-->|"HTTP"| Gradio
    Gradio --> Pipeline
    Pipeline --> Qwen
    Pipeline --> BGE
    Pipeline --> FAISS
    Pipeline --> BM25P
    Pipeline --> Corpus2

    DSApi -.->|"仅数据生成阶段"| Pipeline

    style 用户端 fill:#E3F2FD,stroke:#1565C0
    style 模型 fill:#E1BEE7,stroke:#6A1B9A
    style 存储 fill:#C8E6C9,stroke:#2E7D32
    style 外部服务 fill:#FFE0B2,stroke:#E65100
```
