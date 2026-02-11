# Human-in-the-Loop Tool 구현 가이드

## 📋 핵심 개념

### 1. Config와 Checkpoint 관리

```python
# ✅ 올바른 구조
main_agent_config = {
    "configurable": {
        "thread_id": "user_session_123",  # 메인 대화
        "user_id": "user_abc"              # 사용자 식별
    }
}

tool_graph_config = {
    "configurable": {
        "thread_id": "tool_approval_xyz",  # Tool 내부 워크플로우
        # user_id는 필요시 전달
    }
}
```

**원칙:**
- **분리된 Checkpointer**: 메인 Agent와 Tool 그래프는 각각 독립적인 checkpointer 사용
- **독립적인 thread_id**: Tool 그래프는 자체 thread_id를 생성하여 관리
- **Config 전달 불필요**: Agent → Tool 호출 시 config를 강제로 전달할 필요 없음

---

## 🔄 Human-in-the-Loop 실행 흐름

### 단계별 Flow

```
1. Agent가 Tool 호출
   ↓
2. Tool 그래프 실행 → interrupt_before에서 일시 정지
   ↓
3. Tool이 "승인 대기 중" 메시지 반환
   ↓
4. 승인 ID + Config를 저장 (메모리 또는 DB)
   ↓
5. 사용자에게 승인 요청 UI 표시
   ↓
6. 사용자 응답 (승인/거부)
   ↓
7. update_state()로 상태 갱신
   ↓
8. stream(None, config)로 재개
   ↓
9. 최종 결과 반환
```

---

## 💡 핵심 코드 패턴

### 패턴 1: Interrupt 설정

```python
workflow = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval_node"]  # 이 노드 실행 전 멈춤
)
```

### 패턴 2: 일시 정지까지 실행

```python
config = {"configurable": {"thread_id": "unique_id"}}

# interrupt_before까지 실행
for _ in workflow.stream(initial_state, config, stream_mode="values"):
    pass

# 현재 상태 확인
state = workflow.get_state(config)
print(state.next)  # ['approval_node'] - 다음 실행될 노드
```

### 패턴 3: 상태 업데이트 후 재개

```python
# 사용자 승인 처리
workflow.update_state(config, {"approved": True})

# 재개 (None을 전달하면 기존 상태에서 계속)
for event in workflow.stream(None, config, stream_mode="values"):
    final_state = event
```

---

## ⚠️ 주의사항

### 1. Tool에서 Config 전달 방법

**❌ 잘못된 방법:**
```python
@tool
def my_tool(input: str, config: RunnableConfig) -> str:  # config 직접 받기 불가
    # LangChain의 @tool 데코레이터는 config를 자동 주입하지 않음
    pass
```

**✅ 올바른 방법:**
```python
@tool
def my_tool(input: str) -> str:
    # Tool 내부에서 자체적으로 config 생성
    tool_config = {"configurable": {"thread_id": f"tool_{uuid.uuid4()}"}}
    result = my_graph.invoke(state, config=tool_config)
    return result
```

### 2. Checkpointer 공유 문제

**❌ 안티패턴:**
```python
# 하나의 checkpointer를 여러 그래프에서 공유
shared_checkpointer = MemorySaver()
main_agent = create_agent(checkpointer=shared_checkpointer)
tool_graph = builder.compile(checkpointer=shared_checkpointer)  # 위험!
```

**✅ 권장 패턴:**
```python
# 각 그래프는 독립적인 checkpointer 사용
main_checkpointer = MemorySaver()
tool_checkpointer = MemorySaver()

main_agent = create_agent(checkpointer=main_checkpointer)
tool_graph = builder.compile(checkpointer=tool_checkpointer)
```

### 3. Thread ID 관리

**Production 환경 권장 방식:**

```python
import uuid

# 승인 요청마다 고유 ID 생성
approval_id = str(uuid.uuid4())[:8]
thread_id = f"approval_{approval_id}"

# 데이터베이스에 저장
pending_approvals_db.insert({
    "approval_id": approval_id,
    "thread_id": thread_id,
    "user_id": user_id,
    "config": {"configurable": {"thread_id": thread_id}},
    "created_at": datetime.now()
})
```

---

## 🏭 Production 체크리스트

- [ ] **Persistent Checkpointer 사용**
  - MemorySaver는 개발용
  - PostgresSaver, RedisSaver 등 영구 저장소 사용

- [ ] **승인 요청 저장**
  - 메모리 딕셔너리 대신 DB 사용
  - 만료 시간 설정 (예: 24시간)

- [ ] **에러 처리**
  - 승인 ID 없음
  - 이미 처리된 요청
  - 타임아웃

- [ ] **보안**
  - 승인 요청자와 응답자 일치 확인
  - 권한 검증

- [ ] **모니터링**
  - 대기 중인 승인 수
  - 평균 승인 시간
  - 승인/거부 비율

---

## 📚 참고 자료

### 공식 문서
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
- [Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/)
- [interrupt_before/after](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph.compile)

### 코드 예제
1. `simple_hitl_tool_agent.py` - 기본 데모
2. `production_hitl_tool.py` - Production 패턴

---

## 🎯 요약

| 항목 | 메인 Agent | Tool 그래프 |
|------|------------|-------------|
| **Checkpointer** | MemorySaver (또는 Persistent) | 독립적인 MemorySaver |
| **Thread ID** | 사용자 세션 ID | 승인 요청별 고유 ID |
| **Config 관리** | 외부에서 주입 | 내부에서 생성 |
| **일시 정지** | 불필요 | interrupt_before 사용 |
| **재개 방법** | N/A | update_state + stream(None) |

**핵심**: Tool 그래프는 독립적으로 동작하며, 자체적으로 config와 checkpoint를 관리합니다.
