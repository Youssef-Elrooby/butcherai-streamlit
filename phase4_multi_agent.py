"""
ButcherAI Phase 4 — Multi-Agent Architecture
=============================================
Orchestrator-Worker pattern: one orchestrator agent routes queries to
specialist agents, each with a focused role and tool subset.

Architecture:
  Orchestrator (Router + Planner)
       │
       ├── Inventory Agent  (check_inventory, get_expiry_alerts)
       ├── Sales Agent      (query_sales, calculate_profit_loss)
       └── Procurement Agent (run_forecast, get_procurement_recommendation)

Each specialist has:
  - A focused system prompt
  - A subset of tools
  - Its own conversation memory

Install:
  pip install google-genai

Integration into app.py:
  elif page == "🤖 الوكلاء المتعددين":
      from phase4_multi_agent import show_multi_agent_page
      show_multi_agent_page(db_path=DB, gemini_api_key=GEMINI_API_ENV)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH      = os.environ.get("DB_PATH", "ButcherAI.db")
GEMINI_MODEL = "gemini-flash-lite-latest"

# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST AGENT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
SPECIALIST_AGENTS = {
    "inventory": {
        "name": "Inventory Agent",
        "name_ar": "وكيل المخزون",
        "description": "Manages stock levels, expiry alerts, and reorder recommendations",
        "system_prompt": (
            "You are the Inventory Specialist for an Egyptian butcher shop. "
            "Your focus: stock levels, expiry dates, reorder points, and waste reduction. "
            "You have access to: check_inventory, get_expiry_alerts. "
            "Always provide specific numbers (kg, days remaining). "
            "Answer in the same language the user used (Arabic or English)."
        ),
        "tools": ["check_inventory"],
    },
    "sales": {
        "name": "Sales Agent",
        "name_ar": "وكيل المبيعات",
        "description": "Analyzes sales data, profit/loss, and revenue trends",
        "system_prompt": (
            "You are the Sales Specialist for an Egyptian butcher shop. "
            "Your focus: revenue, profit margins, sales trends, and P&L analysis. "
            "You have access to: query_sales, calculate_profit_loss, get_expenses. "
            "Always provide specific numbers (EGP, kg, percentages). "
            "Answer in the same language the user used (Arabic or English)."
        ),
        "tools": ["query_sales", "calculate_profit_loss", "get_expenses"],
    },
    "procurement": {
        "name": "Procurement Agent",
        "name_ar": "وكيل المشتريات",
        "description": "Handles demand forecasting, buy recommendations, and supplier analysis",
        "system_prompt": (
            "You are the Procurement Specialist for an Egyptian butcher shop. "
            "Your focus: demand forecasting, purchase recommendations, WAC analysis, "
            "and supplier performance. "
            "You have access to: run_forecast, get_procurement_recommendation, "
            "get_wac_analysis, get_supplier_performance, get_holiday_info. "
            "Always provide specific numbers (kg, EGP, days). "
            "Answer in the same language the user used (Arabic or English)."
        ),
        "tools": [
            "run_forecast", "get_procurement_recommendation",
            "get_wac_analysis", "get_supplier_performance", "get_holiday_info",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────────────────────────────────────
ORCHESTRATOR_PROMPT = (
    "You are the Orchestrator for ButcherAI's multi-agent system. "
    "Your role: route user queries to the correct specialist agent(s), "
    "then synthesize their responses into a coherent answer.\n\n"
    "Available specialists:\n"
    "- inventory: Stock levels, expiry alerts, reorder recommendations\n"
    "- sales: Revenue, profit/loss, sales trends, P&L analysis\n"
    "- procurement: Demand forecasting, buy recommendations, supplier analysis\n\n"
    "Routing rules:\n"
    "1. 'كم عندي مخزون؟' → inventory\n"
    "2. 'ما هو صافي الربح؟' → sales\n"
    "3. 'ماذا أشتري؟' → procurement\n"
    "4. 'خطط شراء كامل' → procurement + inventory\n"
    "5. 'تقرير شامل' → all three\n\n"
    "Return ONLY valid JSON:\n"
    '{"agent": "inventory|sales|procurement|all", "reason": "why this agent"}'
)


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────
class SpecialistAgent:
    """
    A specialist agent with a focused role and tool subset.
    Wraps the existing ButcherAIAgent with a custom system prompt and tool filter.
    """

    def __init__(
        self,
        role: str,
        api_key: str,
        db_path: str = DB_PATH,
        ent_model=None,
        ent_features=None,
        get_mult_fn=None,
    ):
        self.role = role
        self.api_key = api_key
        self.db_path = db_path
        self.ent_model = ent_model
        self.ent_features = ent_features
        self.get_mult_fn = get_mult_fn

        config = SPECIALIST_AGENTS.get(role, {})
        self.name = config.get("name", role)
        self.name_ar = config.get("name_ar", role)
        self.system_prompt = config.get("system_prompt", "")
        self.tools = config.get("tools", [])

        self._session = None
        self._session_api_key = None
        self._gemini_tools = None
        self.reasoning_trace: List[Dict] = []

    def _build_tools(self):
        """Build Gemini tool declarations for this specialist's tool subset."""
        from google.genai import types
        from phase2_agent_page import TOOL_SCHEMAS

        self._gemini_tools = [
            types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=schema["parameters"],
                )
            ])
            for schema in TOOL_SCHEMAS
            if schema["name"] in self.tools
        ]

    def run(self, user_message: str) -> Tuple[str, List[Dict]]:
        """
        Execute one turn with this specialist agent.

        Returns:
          answer  : str  — the specialist's response
          trace   : list — reasoning steps
        """
        if not self.api_key:
            return f"API key required for {self.name}", []

        try:
            from google.genai import types
            from gemini_client import AgentChatSession
            from phase2_agent_page import execute_tool
        except ImportError as exc:
            return f"Import error: {exc}", []

        if self._gemini_tools is None:
            self._build_tools()

        if self._session is None or self._session_api_key != self.api_key:
            self._session = AgentChatSession(
                api_key=self.api_key,
                system_instruction=self.system_prompt,
                tools=self._gemini_tools,
                model=GEMINI_MODEL,
            )
            self._session_api_key = self.api_key

        self.reasoning_trace = [{"step": 0, "type": "user_input", "content": user_message}]

        from phase2_agent_page import MAX_TURNS

        final_answer = ""
        step = 0
        next_message = user_message

        while step < MAX_TURNS:
            step += 1

            result = self._session.send(next_message, db_path=self.db_path)
            if result.get("error"):
                final_answer = f"Gemini API error: {result['error']}"
                break

            response = result["response"]
            candidate = response.candidates[0] if response.candidates else None
            if not candidate:
                final_answer = "No response from Gemini."
                break

            tool_calls_made = []
            text_parts = []

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                    fc = part.function_call
                    args = {k: v for k, v in fc.args.items()}
                    tool_calls_made.append({"name": fc.name, "args": args})
                    self.reasoning_trace.append({
                        "step": step, "type": "tool_call",
                        "tool": fc.name, "args": args,
                        "content": f"Calling {fc.name}({json.dumps(args, ensure_ascii=False)[:200]})",
                    })
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text)

            if not tool_calls_made:
                final_answer = "\n".join(text_parts).strip()
                self.reasoning_trace.append({
                    "step": step, "type": "final_answer", "content": final_answer
                })
                break

            # Execute tools
            from google.genai import types
            response_parts = []
            for tc in tool_calls_made:
                t_start = time.time()
                tool_result = execute_tool(
                    tc["name"], tc["args"],
                    db_path=self.db_path,
                    ent_model=self.ent_model,
                    ent_features=self.ent_features,
                    get_mult_fn=self.get_mult_fn,
                    gemini_api_key=self.api_key,
                )
                elapsed = round(time.time() - t_start, 2)
                self.reasoning_trace.append({
                    "step": step, "type": "tool_result",
                    "tool": tc["name"], "result": tool_result,
                    "elapsed": elapsed,
                    "content": json.dumps(tool_result, ensure_ascii=False)[:500],
                })
                response_parts.append(
                    types.Part.from_function_response(name=tc["name"], response=tool_result)
                )

            next_message = response_parts

        return final_answer, self.reasoning_trace


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class OrchestratorAgent:
    """
    Routes user queries to specialist agents and synthesizes their responses.
    """

    def __init__(
        self,
        api_key: str,
        db_path: str = DB_PATH,
        ent_model=None,
        ent_features=None,
        get_mult_fn=None,
    ):
        self.api_key = api_key
        self.db_path = db_path
        self.ent_model = ent_model
        self.ent_features = ent_features
        self.get_mult_fn = get_mult_fn
        self.specialists: Dict[str, SpecialistAgent] = {}
        self.reasoning_trace: List[Dict] = []
        self.blackboard = SharedBlackboard(db_path)
        self.messenger = AgentMessenger(self)

    def _get_specialist(self, role: str) -> SpecialistAgent:
        """Get or create a specialist agent for the given role."""
        if role not in self.specialists:
            self.specialists[role] = SpecialistAgent(
                role=role,
                api_key=self.api_key,
                db_path=self.db_path,
                ent_model=self.ent_model,
                ent_features=self.ent_features,
                get_mult_fn=self.get_mult_fn,
            )
        return self.specialists[role]

    def _route(self, user_message: str) -> Tuple[List[str], str]:
        """
        Determine which specialist agent(s) should handle this query.

        Returns:
          (list of agent roles, routing reason)
        """
        msg = user_message.lower()

        # Simple keyword-based routing
        inventory_keywords = ["مخزون", "stock", "inventory", "صلاحية", "expiry", "reorder"]
        sales_keywords = ["ربح", "profit", "مبيعات", "sales", "إيرادات", "revenue", "p&l"]
        procurement_keywords = ["شراء", "buy", "procurement", "توصيات", "forecast", "مورد", "supplier"]

        needs_inventory = any(kw in msg for kw in inventory_keywords)
        needs_sales = any(kw in msg for kw in sales_keywords)
        needs_procurement = any(kw in msg for kw in procurement_keywords)

        # Determine routing
        if needs_inventory and needs_sales and needs_procurement:
            return ["inventory", "sales", "procurement"], "Full report requested"
        elif needs_inventory and needs_procurement:
            return ["inventory", "procurement"], "Inventory + procurement planning"
        elif needs_sales and needs_procurement:
            return ["sales", "procurement"], "Sales analysis + procurement"
        elif needs_inventory:
            return ["inventory"], "Inventory query"
        elif needs_sales:
            return ["sales"], "Sales query"
        elif needs_procurement:
            return ["procurement"], "Procurement query"
        else:
            # Default: ask all for a comprehensive answer
            return ["inventory", "sales", "procurement"], "General query — all agents"

    def run(self, user_message: str) -> Tuple[str, List[Dict]]:
        """
        Route the query to specialist agents and synthesize their responses.

        Returns:
          final_answer  : str  — synthesized answer
          trace         : list — full reasoning trace
        """
        if not self.api_key:
            return "Gemini API key required for multi-agent system.", []

        self.reasoning_trace = [{"step": 0, "type": "user_input", "content": user_message}]

        # ── Phase 1: Route the query ──────────────────────────────────────────
        agents_to_use, routing_reason = self._route(user_message)
        self.reasoning_trace.append({
            "step": 1, "type": "routing",
            "content": f"Routed to: {', '.join(agents_to_use)} — {routing_reason}",
        })

        # ── Phase 2: Execute specialist agents ───────────────────────────────
        import uuid
        session_id = str(uuid.uuid4())[:8]
        specialist_results = {}
        for role in agents_to_use:
            specialist = self._get_specialist(role)
            self.reasoning_trace.append({
                "step": 2, "type": "specialist_start",
                "content": f"Starting {specialist.name}...",
            })

            # Check if another agent already wrote useful data to the blackboard
            blackboard_context = ""
            prior_findings = self.blackboard.read_all(session_id)
            if prior_findings:
                blackboard_context = "\n\nPrior findings from other agents:\n"
                for f in prior_findings:
                    blackboard_context += f"- {f['agent']}: {f['key']} = {json.dumps(f['value'], ensure_ascii=False)[:200]}\n"

            # Run specialist with blackboard context
            answer, trace = specialist.run(user_message + blackboard_context)

            # Write findings to blackboard
            self.blackboard.write(role, f"{role}_status", {
                "answer": answer[:500],
                "tools_used": [s.get("tool") for s in trace if s.get("type") == "tool_call"],
            }, session_id)

            specialist_results[role] = {
                "name": specialist.name,
                "name_ar": specialist.name_ar,
                "answer": answer,
                "trace": trace,
            }

            self.reasoning_trace.append({
                "step": 2, "type": "specialist_complete",
                "content": f"{specialist.name} completed",
                "answer_preview": answer[:200],
            })

        # ── Phase 3: Agent-to-Agent Communication ────────────────────────────
        # If procurement needs inventory data, let them communicate directly
        if "procurement" in agents_to_use and "inventory" in agents_to_use:
            self.reasoning_trace.append({
                "step": 3, "type": "agent_communication",
                "content": "Procurement Agent requesting inventory data from Inventory Agent...",
            })
            import uuid
            session_id = str(uuid.uuid4())[:8]
            inventory_data = self.messenger.send_request(
                from_agent="procurement",
                to_agent="inventory",
                request="What is the current stock level and reorder status?",
                session_id=session_id,
            )
            self.reasoning_trace.append({
                "step": 3, "type": "agent_communication_response",
                "content": f"Inventory Agent responded: {inventory_data[:200]}",
            })

        # ── Phase 4: Synthesize ──────────────────────────────────────────────
        self.reasoning_trace.append({
            "step": 4, "type": "synthesizing",
            "content": "Synthesizing specialist responses...",
        })

        synthesis_prompt = (
            f"User query: {user_message}\n\n"
            f"Routing: {routing_reason}\n\n"
        )
        for role, result in specialist_results.items():
            synthesis_prompt += (
                f"--- {result['name']} ({role}) ---\n"
                f"{result['answer']}\n\n"
            )

        synthesis_prompt += (
            "Synthesize the above specialist responses into a single coherent answer. "
            "Remove redundancy, resolve conflicts, and present a unified view. "
            "Answer in the same language the user used (Arabic or English). "
            "Be specific with numbers. Do not add new information."
        )

        try:
            from gemini_client import AgentChatSession
            synth_session = AgentChatSession(
                api_key=self.api_key,
                system_instruction="You are a synthesis assistant for a multi-agent system.",
                model=GEMINI_MODEL,
                tools=None,
            )
            synth_result = synth_session.send(synthesis_prompt)
            if synth_result.get("error"):
                final_answer = f"Synthesis error: {synth_result['error']}"
            else:
                response = synth_result["response"]
                candidate = response.candidates[0] if response.candidates else None
                if candidate:
                    final_answer = ""
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_answer += part.text
                    final_answer = final_answer.strip()
                else:
                    final_answer = "No response from synthesis."
        except Exception as exc:
            final_answer = f"Synthesis failed: {exc}"

        self.reasoning_trace.append({
            "step": 4, "type": "final_answer",
            "content": final_answer,
        })

        return final_answer, self.reasoning_trace


# ─────────────────────────────────────────────────────────────────────────────
# SHARED BLACKBOARD  — Exercise 3.3
# ─────────────────────────────────────────────────────────────────────────────
class SharedBlackboard:
    """
    A shared SQLite-based blackboard where agents write findings and read
    from each other. This enables parallel agent execution and coordination.

    Table schema:
      id          INTEGER PRIMARY KEY
      agent       TEXT    — which agent wrote this
      key         TEXT    — finding key (e.g., "inventory_status")
      value       TEXT    — finding value (JSON)
      timestamp   TEXT    — when written
      session_id  TEXT    — groups findings from the same user query
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Create the blackboard table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_blackboard (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent       TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                session_id  TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blackboard_session ON agent_blackboard(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_blackboard_key ON agent_blackboard(key)"
        )
        conn.commit()
        conn.close()

    def write(self, agent: str, key: str, value: Any, session_id: str):
        """Write a finding to the blackboard."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO agent_blackboard (agent, key, value, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (agent, key, json.dumps(value, ensure_ascii=False), datetime.now().isoformat(), session_id),
        )
        conn.commit()
        conn.close()

    def read(self, key: str, session_id: str) -> Optional[Dict]:
        """Read the latest finding for a given key."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT agent, value, timestamp FROM agent_blackboard "
            "WHERE key = ? AND session_id = ? ORDER BY timestamp DESC LIMIT 1",
            (key, session_id),
        ).fetchone()
        conn.close()
        if row:
            return {
                "agent": row[0],
                "value": json.loads(row[1]),
                "timestamp": row[2],
            }
        return None

    def read_all(self, session_id: str) -> List[Dict]:
        """Read all findings for a session."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT agent, key, value, timestamp FROM agent_blackboard "
            "WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        conn.close()
        return [
            {
                "agent": r[0],
                "key": r[1],
                "value": json.loads(r[2]),
                "timestamp": r[3],
            }
            for r in rows
        ]

    def clear_session(self, session_id: str):
        """Clear all findings for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "DELETE FROM agent_blackboard WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT-TO-AGENT COMMUNICATION  — Exercise 3.4
# ─────────────────────────────────────────────────────────────────────────────
class AgentMessenger:
    """
    Enables direct agent-to-agent communication.
    When one agent needs data from another, it sends a request via the messenger
    instead of going through the user.

    This is the key pattern for multi-agent systems:
    - Procurement Agent needs sales data → asks Sales Agent directly
    - Inventory Agent needs forecast → asks Procurement Agent directly
    """

    def __init__(self, orchestrator: OrchestratorAgent):
        self.orchestrator = orchestrator
        self.message_log: List[Dict] = []

    def send_request(
        self,
        from_agent: str,
        to_agent: str,
        request: str,
        session_id: str,
    ) -> str:
        """
        Send a request from one agent to another.

        Args:
            from_agent: The requesting agent's role
            to_agent: The target agent's role
            request: What data is needed
            session_id: Current session ID

        Returns:
            The target agent's response
        """
        self.message_log.append({
            "from": from_agent,
            "to": to_agent,
            "request": request,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
        })

        # Get the target specialist
        target = self.orchestrator._get_specialist(to_agent)

        # Construct a focused request
        focused_request = (
            f"Another agent ({from_agent}) needs the following information. "
            f"Please provide a concise response with specific data.\n\n"
            f"Request: {request}"
        )

        # Run the target agent
        answer, _ = target.run(focused_request)

        self.message_log.append({
            "from": to_agent,
            "to": from_agent,
            "response": answer[:500],
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
        })

        return answer

    def get_message_log(self, session_id: str) -> List[Dict]:
        """Get all messages for a session."""
        return [m for m in self.message_log if m.get("session_id") == session_id]


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_multi_agent_page(
    db_path: str = DB_PATH,
    gemini_api_key: str = "",
    ent_model=None,
    ent_features=None,
    get_mult_fn=None,
):
    """Full Streamlit page for the multi-agent system."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.title("🤖 الوكلاء المتعددين (Multi-Agent)")
    st.markdown(
        "<div style='background:#d1ecf1;border:2px solid #17a2b8;border-radius:8px;"
        "padding:10px 14px;margin:6px 0;font-size:14px'>"
        "النظام يوزع السؤال على وكلاء متخصصين: مخزون، مبيعات، مشتريات — "
        "ثم يجمع إجاباتهم في إجابة واحدة متماسكة."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = st.text_input(
        "🔑 Gemini API Key (مطلوب):",
        type="password",
        value=gemini_api_key,
    )

    # ── Initialize orchestrator ───────────────────────────────────────────────
    if "multi_agent_obj" not in st.session_state or api_key != getattr(
        st.session_state.get("multi_agent_obj"), "api_key", ""
    ):
        st.session_state["multi_agent_obj"] = OrchestratorAgent(
            api_key=api_key, db_path=db_path,
            ent_model=ent_model, ent_features=ent_features,
            get_mult_fn=get_mult_fn,
        )
        st.session_state["multi_agent_history"] = []

    orchestrator: OrchestratorAgent = st.session_state["multi_agent_obj"]

    # ── Example prompts ───────────────────────────────────────────────────────
    st.subheader("💬 أمثلة")
    examples = [
        "تقرير شامل عن المحل",
        "كم مخزوني وما الذي يجب أن أشتريه؟",
        "حلل المبيعات وأعطني توصيات شراء",
        "ما هو صافي الربح ووضع المخزون؟",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(f"💡 {ex}", key=f"ma_ex_{i}"):
            st.session_state["ma_pending"] = ex

    st.divider()

    # ── Conversation display ──────────────────────────────────────────────────
    for turn in st.session_state.get("multi_agent_history", []):
        role = turn["role"]
        icon = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=icon):
            st.markdown(turn["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    pending = st.session_state.pop("ma_pending", None)
    user_msg = st.chat_input("اسأل النظام المتعدد الوكلاء...") or pending

    if user_msg:
        st.session_state["multi_agent_history"].append({"role": "user", "content": user_msg})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_msg)

        with st.chat_message("assistant", avatar="🤖"):
            reasoning_placeholder = st.empty()
            answer_placeholder = st.empty()

            if not api_key:
                answer_placeholder.error("❌ أدخل Gemini API Key لتشغيل النظام.")
            else:
                with st.spinner("🧠 الوكلاء يعملون..."):
                    t_start = time.time()
                    orchestrator.api_key = api_key
                    answer, trace = orchestrator.run(user_msg)
                    elapsed = round(time.time() - t_start, 2)

                # Display routing + specialist results
                routing_steps = [s for s in trace if s["type"] == "routing"]
                specialist_steps = [s for s in trace if s["type"] == "specialist_complete"]

                with reasoning_placeholder.expander(
                    f"🔍 التوجيه والوكلاء ({len(specialist_steps)} وكلاء، {elapsed}s)",
                    expanded=False,
                ):
                    for s in trace:
                        if s["type"] == "routing":
                            st.info(f"📋 {s['content']}")
                        elif s["type"] == "specialist_start":
                            st.markdown(f"▶️ {s['content']}")
                        elif s["type"] == "specialist_complete":
                            st.markdown(f"✅ {s['content']}")
                            if "answer_preview" in s:
                                st.caption(s["answer_preview"][:150])
                        elif s["type"] == "synthesizing":
                            st.info(f"🔄 {s['content']}")

                answer_placeholder.markdown(answer)

        st.session_state["multi_agent_history"].append({"role": "assistant", "content": answer})

    # ── Sidebar: specialist registry ──────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("**🛠️ الوكلاء المتخصصون:**")
        for role, config in SPECIALIST_AGENTS.items():
            st.markdown(f"• **{config['name_ar']}** — {config['description']}")
