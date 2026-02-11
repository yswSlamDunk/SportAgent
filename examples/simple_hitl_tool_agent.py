"""
가장 간단한 Human-in-the-Loop Tool + Agent 예제

구조:
1. 승인이 필요한 Tool 그래프 (human-in-the-loop 포함)
2. 해당 Tool을 사용하는 외부 Agent
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import operator


# ============================================
# 1. Human-in-the-Loop를 포함하는 Tool 그래프
# ============================================

class ApprovalState(TypedDict):
    """승인 요청 상태"""
    request: str  # 승인 요청 내용
    approved: bool  # 승인 여부
    result: str  # 최종 결과


def request_approval(state: ApprovalState) -> dict:
    """승인 요청 - 여기서 일시 정지됨"""
    print(f"\n🔔 승인 요청: {state['request']}")
    print("👉 승인 대기 중... (interrupt_before='approve_node')")
    return {}


def approve_node(state: ApprovalState) -> dict:
    """승인 처리 노드"""
    if state.get("approved"):
        result = f"✅ 승인됨! 요청 '{state['request']}' 처리 완료"
    else:
        result = f"❌ 거부됨. 요청 '{state['request']}' 취소됨"
    
    print(f"\n{result}")
    return {"result": result}


# 승인 워크플로우 그래프 생성
def create_approval_workflow():
    """Human-in-the-loop 워크플로우 생성"""
    builder = StateGraph(ApprovalState)
    
    builder.add_node("request_approval", request_approval)
    builder.add_node("approve_node", approve_node)
    
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", "approve_node")
    builder.add_edge("approve_node", END)
    
    # 핵심: approve_node 실행 전에 일시 정지
    checkpointer = MemorySaver()
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["approve_node"]  # 이 노드 실행 전 일시 정지
    )


# Tool로 만들기 위한 래퍼 함수
approval_graph = create_approval_workflow()


@tool
def request_database_change(request: str) -> str:
    """
    데이터베이스 변경을 요청합니다. 사용자 승인이 필요합니다.
    
    Args:
        request: 변경 요청 내용 (예: "사용자 테이블에서 ID 123 삭제")
    
    Returns:
        승인 결과 메시지
    """
    # Tool 전용 thread_id 생성
    tool_thread_id = f"tool_{hash(request) % 10000}"
    config = {"configurable": {"thread_id": tool_thread_id}}
    
    initial_state = {
        "request": request,
        "approved": False,
        "result": ""
    }
    
    print(f"\n{'='*60}")
    print(f"🔧 Tool 실행: request_database_change")
    print(f"📝 요청 내용: {request}")
    print(f"🔑 Tool Thread ID: {tool_thread_id}")
    print(f"{'='*60}")
    
    # 1단계: 승인 대기까지 실행
    for event in approval_graph.stream(initial_state, config, stream_mode="values"):
        pass  # interrupt_before에서 자동으로 멈춤
    
    # 현재 상태 확인
    current_state = approval_graph.get_state(config)
    print(f"\n⏸️  일시 정지됨!")
    print(f"   다음 노드: {current_state.next}")
    print(f"   Checkpoint ID: {current_state.config['configurable']['checkpoint_id']}")
    
    # 사용자 입력 대기 (실제로는 여기서 API로 중단되고, 나중에 재개됨)
    print(f"\n{'='*60}")
    user_input = input("✋ 승인하시겠습니까? (y/n): ").strip().lower()
    print(f"{'='*60}")
    
    # 상태 업데이트: 승인 여부를 반영
    approval_graph.update_state(
        config,
        {"approved": user_input == 'y'}
    )
    
    # 2단계: 재개하여 끝까지 실행
    print("\n▶️  그래프 재개...")
    final_state = None
    for event in approval_graph.stream(None, config, stream_mode="values"):
        final_state = event
    
    return final_state["result"]


# ============================================
# 2. 외부 Agent (Tool 사용자)
# ============================================

def create_main_agent():
    """Human-in-the-loop tool을 사용하는 메인 에이전트"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    tools = [request_database_change]
    
    # checkpointer 포함하여 에이전트 생성
    checkpointer = MemorySaver()
    agent = create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer
    )
    
    return agent


# ============================================
# 3. 실행 예제
# ============================================

def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🤖 Human-in-the-Loop Tool을 사용하는 Agent 데모")
    print("="*60 + "\n")
    
    # 에이전트 생성
    agent = create_main_agent()
    
    # 메인 에이전트의 config (사용자별 대화 관리)
    main_config = {
        "configurable": {
            "thread_id": "user_conversation_001"
        }
    }
    
    # 사용자 요청
    user_message = "데이터베이스에서 사용자 ID 12345를 삭제해줘"
    
    print(f"👤 사용자: {user_message}\n")
    
    # 에이전트 실행
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=main_config
    )
    
    print(f"\n{'='*60}")
    print("🤖 최종 응답:")
    print(f"{'='*60}")
    print(response["messages"][-1].content)
    
    print(f"\n{'='*60}")
    print("✅ 완료!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
