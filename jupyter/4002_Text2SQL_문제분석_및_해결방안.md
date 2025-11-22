# Text2SQL 노트북 문제 분석 및 해결 방안

## 문제점 분석

### 1. 주요 문제: SQL 쿼리 생성 시 Tool Call 사용

**현상:**
- `query_gen_node`에서 LLM이 SQL 쿼리를 생성할 때 `db_query_tool`을 tool call로 호출하려고 시도
- 에러 메시지: "The wrong tool was called: db_query_tool. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call."

**원인:**
```python
query_gen = query_gen_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).bind_tools([SubmitFinalAnswer, db_query_tool])  # ❌ 문제: db_query_tool이 bind되어 있음
```

SQL 쿼리 생성 단계에서는 tool call을 사용하지 않고 직접 텍스트로 출력해야 하는데, `db_query_tool`이 bind되어 있어 LLM이 tool call을 시도합니다.

### 2. 워크플로우 엣지 문제

**현상:**
- `query_generate` 노드에서 여러 엣지가 동시에 정의되어 있어 경로가 모호함
- `query_generate -> human_in_the_loop`와 `query_generate -> END`가 동시에 존재

**원인:**
```python
workflow.add_conditional_edges("query_generate", should_continue)  # 조건부 엣지
workflow.add_edge("query_generate", "human_in_the_loop")  # ❌ 중복 엣지
workflow.add_edge("query_generate", END)  # ❌ 중복 엣지
```

조건부 엣지와 일반 엣지가 충돌합니다.

### 3. should_continue 함수 로직 문제

**현상:**
- SQL 쿼리 텍스트를 제대로 감지하지 못함
- ToolMessage 처리 로직 부재

**원인:**
```python
def should_continue(state: SqlState):
    last_message = messages[-1]
    if last_message.content.startswith("Answer:"):  # ❌ ToolMessage인 경우 처리 안됨
        return END
    # ...
```

### 4. 무한 루프 가능성

**현상:**
- 에러가 발생해도 계속 재시도하여 recursion limit에 도달

**원인:**
- 에러 처리 후 재시도 로직이 명확하지 않음
- 쿼리 검증 실패 시 적절한 종료 조건 없음

## 해결 방안

### 해결책 1: SQL 쿼리 생성 시 Tool Call 제거

```python
# 수정 전
query_gen = query_gen_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).bind_tools([SubmitFinalAnswer, db_query_tool])  # ❌

# 수정 후
query_gen = query_gen_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
)  # ✅ tool을 bind하지 않음 - SQL 쿼리를 직접 텍스트로 출력
```

**이유:**
- SQL 쿼리 생성 단계에서는 LLM이 tool call 없이 직접 SQL 텍스트를 출력해야 함
- `model_check_query` 노드에서 이 텍스트를 추출하여 `db_query_tool`을 호출

### 해결책 2: query_gen_node 수정

```python
def query_gen_node(state: SqlState):
    """
    SQL 쿼리를 생성하는 노드.
    LLM은 SQL 쿼리를 직접 텍스트로 출력해야 함 (tool call 없이).
    """
    message = query_gen.invoke(state)
    
    # tool call이 있는 경우는 에러 처리
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            tool_messages.append(
                ToolMessage(
                    content="Error: SQL queries must be generated as plain text, not as tool calls. Please output the SQL query directly in your response.",
                    tool_call_id=tc["id"],
                )
            )
    
    return {"messages": [message] + tool_messages}
```

### 해결책 3: should_continue 함수 개선

```python
def should_continue(state: SqlState) -> Literal[END, "model_check", "query_generate", "human_in_the_loop"]:
    """
    query_generate 노드에서 나온 결과를 기반으로 다음 노드를 결정.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # ToolMessage인 경우, 이전 AIMessage를 확인
    if isinstance(last_message, ToolMessage) and len(messages) > 1:
        prev_message = messages[-2]
        content = prev_message.content if hasattr(prev_message, 'content') else ""
    else:
        content = last_message.content if hasattr(last_message, 'content') else ""
    
    if content.startswith("Answer:"):
        return END
    elif content.startswith("No Result:"):
        return "human_in_the_loop"
    elif content.startswith("Error:"):
        return "query_generate"
    else:
        # SQL 쿼리 텍스트인 경우 model_check로 이동
        return "model_check"
```

### 해결책 4: 워크플로우 엣지 정리

```python
workflow = StateGraph(SqlState)

workflow.add_node("sql_query_rewrite", query_rewrite_sql)
workflow.add_node("query_generate", query_gen_node)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("model_check", model_check_query)
workflow.add_node('human_in_the_loop', human_in_the_loop)

workflow.add_edge(START, "sql_query_rewrite")
workflow.add_edge("sql_query_rewrite", "query_generate")

# query_generate에서 나온 결과에 따라 분기 (조건부 엣지만 사용)
workflow.add_conditional_edges(
    "query_generate",
    should_continue,
    {
        END: END,
        "model_check": "model_check",
        "query_generate": "query_generate",
        "human_in_the_loop": "human_in_the_loop"
    }
)

workflow.add_edge("model_check", "execute_query")
workflow.add_edge("execute_query", "query_generate")
workflow.add_edge("human_in_the_loop", "query_generate")

app = workflow.compile(checkpointer=MemorySaver())
```

**변경 사항:**
- `query_generate`에서 나가는 일반 엣지 제거
- 조건부 엣지만 사용하여 명확한 경로 정의

### 해결책 5: model_check_query 개선

```python
def model_check_query(state: SqlState) -> dict[str, list[AIMessage]]:
    """
    SQL 쿼리를 검증하고 db_query_tool을 호출하는 노드.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # SQL 쿼리 추출 (마지막 AIMessage의 content에서)
    sql_query = ""
    if hasattr(last_message, 'content') and last_message.content:
        content = last_message.content.strip()
        # SELECT로 시작하는 부분 추출
        if "SELECT" in content.upper():
            start_idx = content.upper().find("SELECT")
            if start_idx != -1:
                sql_query = content[start_idx:].strip()
                if sql_query.endswith(';'):
                    sql_query = sql_query[:-1]
    
    if not sql_query:
        error_msg = AIMessage(content="Error: Could not extract SQL query from the response. Please generate a valid SQL query.")
        return {"messages": [error_msg]}
    
    # query_check를 통해 쿼리 검증 및 db_query_tool 호출
    tmp = query_check.invoke({"messages": [last_message]})
    
    return {"messages": [tmp]}
```

## 권장 워크플로우

```
START 
  → sql_query_rewrite (질의 재작성)
  → query_generate (SQL 쿼리 생성 - 텍스트로 출력)
    ├─ "Answer:" → END
    ├─ "No Result:" → human_in_the_loop → query_generate
    ├─ "Error:" → query_generate (재시도)
    └─ SQL 쿼리 텍스트 → model_check (쿼리 검증)
      → execute_query (쿼리 실행)
      → query_generate (결과 처리)
```

## 추가 개선 사항

1. **프롬프트 개선**: SQL 쿼리를 텍스트로 출력하도록 명확히 지시
2. **에러 처리 강화**: 재시도 횟수 제한 및 명확한 에러 메시지
3. **로깅 개선**: 각 단계에서 디버깅 정보 출력

