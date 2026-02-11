"""
Production Ready: Agent + Human-in-the-Loop Tool Integration
(Derived ID & Asynchronous Approval Pattern)

이 코드는 다음 요구사항을 충족합니다:
1. Agent가 Tool을 보유하고 실행합니다.
2. Tool 내부는 LangGraph로 구현되어 있으며, interrupt(HITL)가 발생합니다.
3. Tool은 Agent에게 "대기 중"임을 알리고 종료되며, 실제 승인은 비동기적으로 처리됩니다.
"""

import uuid
import operator
from typing import TypedDict, Annotated, List, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


# ============================================
# 1. [Inner] 승인 워크플로우 (The Tool's Logic)
# ============================================

class ApprovalState(TypedDict):
    request: str
    approved: Optional[bool]
    final_output: str

def request_step(state: ApprovalState):
    """승인 요청 단계"""
    # 실제로는 이곳에서 슬랙/이메일 알림 등을 보낼 수 있음
    return {"final_output": f"Requesting approval for: {state['request']}"}

def execution_step(state: ApprovalState):
    """승인 후 실행 단계"""
    if state.get("approved"):
        # 실제 DB 작업 수행
        result = f"✅ SUCCESS: Executed '{state['request']}'"
    else:
        result = f"❌ DENIED: Operation '{state['request']}' was rejected."
    
    return {"final_output": result}

# Tool 내부의 그래프 정의
_approval_builder = StateGraph(ApprovalState)
_approval_builder.add_node("request", request_step)
_approval_builder.add_node("execution", execution_step)
_approval_builder.add_edge(START, "request")
_approval_builder.add_edge("request", "execution")
_approval_builder.add_edge("execution", END)

# Tool 내부 전용 Checkpointer
# 주의: 실제 배포시에는 PostgresSaver 등 영구 저장소 사용 필수
_inner_checkpointer = MemorySaver()

# interrupt_before='execution'으로 설정하여 실행 전 멈춤
approval_graph = _approval_builder.compile(
    checkpointer=_inner_checkpointer,
    interrupt_before=["execution"]
)


# ============================================
# 2. [Outer] Agent가 사용할 Tool 정의
# ============================================

# 승인 대기열 (DB 대용)
pending_requests = {}

@tool
def sensitive_action_tool(
    action_description: str, 
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    민감한 작업을 수행하는 도구입니다. 
    이 도구는 즉시 실행되지 않고 승인 요청을 생성합니다.
    
    Args:
        action_description: 수행할 작업에 대한 설명
        config: LangChain Context (자동 주입)
    """
    # 1. Main Agent의 Thread ID 가져오기 (Derived ID 생성을 위해)
    parent_config = config.get("configurable", {})
    parent_thread_id = parent_config.get("thread_id", "default")
    
    # 2. Tool을 위한 고유 Thread ID 생성 (Derived ID)
    # 포맷: {부모ID}_tool_{UUID}
    action_id = str(uuid.uuid4())[:8]
    tool_thread_id = f"{parent_thread_id}_tool_{action_id}"
    
    tool_config = {
        "configurable": {
            "thread_id": tool_thread_id,
            # 메타데이터로 부모 정보 남기기
            "parent_id": parent_thread_id 
        }
    }
    
    print(f"\n[Tool] 🚀 Starting logic for action: {action_description}")
    print(f"[Tool] 🔗 Thread Linking: {parent_thread_id} -> {tool_thread_id}")

    # 3. Inner Graph 실행 (Interrupt 지점까지)
    initial_state = {
        "request": action_description,
        "approved": None,
        "final_output": ""
    }
    
    # stream을 사용하여 실행 (일시정지 지점에서 멈춤)
    # invoke()를 쓰면 interrupt시 에러가 발생하거나 리턴값이 다를 수 있어 stream 권장
    final_output = "No output"
    
    for event in approval_graph.stream(initial_state, tool_config, stream_mode="values"):
        if "final_output" in event:
            final_output = event["final_output"]

    # 4. 상태 확인 (정말로 멈췄는지)
    state_snapshot = approval_graph.get_state(tool_config)
    
    if state_snapshot.next:
        # 멈춘 상태 (승인 대기)
        print(f"[Tool] ⏸️  Paused before: {state_snapshot.next}")
        
        # 대기열에 등록 (외부 시스템 연동용)
        pending_requests[action_id] = {
            "tool_thread_id": tool_thread_id,
            "request": action_description,
            "parent_thread_id": parent_thread_id
        }
        
        return (
            f"⚠️ 승인 요청이 생성되었습니다. (ID: {action_id})\n"
            f"관리자가 승인할 때까지 작업은 대기 상태입니다.\n"
            f"상태: Pending Approval"
        )
    else:
        # 멈추지 않고 끝났다면 (혹은 에러)
        return f"Tool execution finished unexpected: {final_output}"


# ============================================
# 3. [System] 승인 처리 시스템 (API 시뮬레이션)
# ============================================

def admin_approve_action(action_id: str, approved: bool):
    """관리자가 승인 버튼을 눌렀을 때 호출되는 함수"""
    print(f"\n👮‍♂️ [Admin] Processing approval for ID: {action_id} (Approved: {approved})")
    
    if action_id not in pending_requests:
        print("❌ Error: Invalid Action ID")
        return

    req_info = pending_requests[action_id]
    tool_thread_id = req_info["tool_thread_id"]
    
    # 해당 Tool Thread의 설정 복원
    tool_config = {"configurable": {"thread_id": tool_thread_id}}
    
    # 1. 상태 업데이트 (승인/거부 결정 주입)
    approval_graph.update_state(tool_config, {"approved": approved})
    
    print("[Admin] ▶️  Resuming tool execution...")
    
    # 2. 실행 재개 (None을 입력하여 멈춘 곳부터 계속)
    final_result = None
    for event in approval_graph.stream(None, tool_config, stream_mode="values"):
        final_result = event
        
    print(f"[Admin] ✅ Final Tool Output: {final_result.get('final_output', 'Unknown')}")
    
    # 대기열에서 제거
    del pending_requests[action_id]


# ============================================
# 4. [Main] 에이전트 생성 및 전체 시나리오
# ============================================

def main():
    # 1. 메인 에이전트 설정
    # NOTE: 실제 사용시에는 OpenAI Key 설정 필요
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # 에이전트에게 Checkpointer 부여 (대화 기억용)
    agent_checkpointer = MemorySaver()
    
    agent = create_react_agent(
        llm, 
        tools=[sensitive_action_tool],
        checkpointer=agent_checkpointer
    )

    # 2. 사용자 시나리오 시작
    user_thread_id = "user_session_001"
    agent_config = {"configurable": {"thread_id": user_thread_id}}
    
    print("="*60)
    print("🤖 Agent Scenario Start")
    print("="*60)
    
    query = "운영 DB에서 'users' 테이블을 삭제해줘. 이건 긴급 요청이야."
    print(f"👤 User: {query}")
    
    # Agent 실행
    response = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=agent_config
    )
    
    # Agent의 마지막 응답 확인
    last_msg = response["messages"][-1].content
    print(f"\n🤖 Agent Response:\n{last_msg}")
    
    # --- 비동기 상황 시뮬레이션 ---
    
    print("\n" + "-"*60)
    print("🕒 (Time passes... User waits for approval)")
    print("-"*60)
    
    # 현재 대기 중인 요청 확인
    if not pending_requests:
        print("No pending requests.")
        return

    # 대기 중인 첫 번째 요청 가져오기
    target_id = list(pending_requests.keys())[0]
    req_data = pending_requests[target_id]
    
    print(f"📋 Pending Request Found: {req_data['request']} (ID: {target_id})")
    print(f"   -> Linked to Agent Thread: {req_data['parent_thread_id']}")
    
    # 관리자 승인 (외부 트리거)
    admin_approve_action(target_id, approved=True)
    
    # (선택사항) Agent에게 결과를 알려주고 싶다면?
    # 방법 1: 사용자가 "승인됐어 확인해봐"라고 다시 말한다.
    # 방법 2: 시스템이 Agent에게 도구 출력을 주입한다. (고급)
    
    print("\n" + "="*60)
    print("🤖 Agent Scenario Continue")
    print("="*60)
    
    follow_up = "방금 그 요청 승인되었어. 확인해줄래?"
    print(f"👤 User: {follow_up}")
    
    # 여기서는 간단히 대화를 이어가서 Agent가 기억하는지 확인
    # (참고: Agent는 도구가 '대기 상태'로 끝났다는 것만 알고, 
    #  실제 내부 작업이 완료되었는지는 문맥상으로만 알 수 있음)
    response_2 = agent.invoke(
        {"messages": [HumanMessage(content=follow_up)]},
        config=agent_config
    )
    
    print(f"\n🤖 Agent Response:\n{response_2['messages'][-1].content}")


if __name__ == "__main__":
    main()
