import os
import streamlit as st
import streamlit.components.v1 as components

BROWSER_WS_PORT = os.getenv("BACKEND_WS_PORT", "8000")

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")


def model_name(model_type):
    return "Fine-tuned Qwen (QLoRA)" if model_type == "finetuned" else "Base Qwen (Qwen2.5-1.5B-Instruct)"


def render_browser_chat(selected_model):
    """Actual chat WebSocket runs in the browser, so Chrome DevTools can see /ws/agent."""
    model_label = model_name(selected_model).replace("\\", "\\\\").replace('"', '\\"')
    components.html(f'''<!doctype html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;background:transparent;color:#f5f5f5;font-family:Arial,sans-serif}}
#connection{{font-size:13px;color:#aaa;margin:4px 0 10px}}
#messages{{min-height:180px;max-height:620px;overflow:auto;padding:4px 0 10px}}
.row{{display:flex;margin:10px 0}}.user-row{{justify-content:flex-end}}.assistant-row{{justify-content:flex-start}}
.bubble{{max-width:92%;padding:13px 16px;border-radius:12px;line-height:1.6;white-space:pre-wrap;word-break:break-word}}
.user{{background:#2b2d42}}.assistant{{background:#171922;border:1px solid #2b2e3b}}
.meta,.sources{{margin-top:9px;color:#999;font-size:12px;white-space:pre-wrap;line-height:1.6}}
#status{{font-size:12px;color:#aaa;min-height:18px;margin:4px 0}}
#composer{{display:flex;gap:8px;margin-top:5px}}#question{{flex:1;resize:none;min-height:48px;max-height:150px;padding:13px;border-radius:10px;border:1px solid #383b49;background:#292b35;color:#fff;outline:none;font-size:15px}}
#send{{width:52px;border:0;border-radius:10px;background:#3b3d4a;color:#fff;cursor:pointer;font-size:20px}}#send:disabled{{opacity:.5;cursor:not-allowed}}
</style></head><body>
<div id="connection">🔌 Connecting...</div><div id="messages"></div><div id="status"></div>
<div id="composer"><textarea id="question" placeholder="Ask something..." rows="1"></textarea><button id="send" disabled>↑</button></div>
<script>
(() => {{
 const PORT="{BROWSER_WS_PORT}"; const MODEL="{selected_model}"; const LABEL="{model_label}";
 const conn=document.getElementById("connection"), msgs=document.getElementById("messages"), status=document.getElementById("status"), q=document.getElementById("question"), send=document.getElementById("send");
 let ws=null, reconnectTimer=null, current=null, answer="", tools=[], sources=[], prompt="", backendModel="";
 function url() {{ const host=location.hostname||"127.0.0.1"; const proto=location.protocol==="https:"?"wss:":"ws:"; return proto+"//"+host+":"+PORT+"/ws/agent"; }}
 function scroll() {{ msgs.scrollTop=msgs.scrollHeight; }}
 function startAssistant() {{
   const row=document.createElement("div"); row.className="row assistant-row";
   const bubble=document.createElement("div"); bubble.className="bubble assistant";
   const content=document.createElement("div"), meta=document.createElement("div"), src=document.createElement("div");
   meta.className="meta"; src.className="sources"; bubble.append(content,meta,src); row.appendChild(bubble); msgs.appendChild(row);
   current={{content,meta,src}}; answer=""; tools=[]; sources=[]; prompt=""; backendModel=""; scroll();
 }}
 function update() {{
   if(!current) startAssistant(); current.content.textContent=answer;
   let m="🤖 Model: "+(backendModel||LABEL); if(tools.length)m+="\\n🛠️ Tools used: "+tools.join(", "); if(prompt)m+="\\n📝 Prompt version: "+prompt;
   current.meta.textContent=m; current.src.textContent=sources.length?"📚 Sources\\n"+sources.join("\\n"):""; scroll();
 }}
 function reconnect() {{ if(reconnectTimer)return; reconnectTimer=setTimeout(()=>{{reconnectTimer=null; connect();}},1000); }}
 function connect() {{
   conn.textContent="🔌 Connecting to "+url(); send.disabled=true;
   try {{ ws=new WebSocket(url()); }} catch(e) {{ conn.textContent="🔴 WebSocket creation failed"; reconnect(); return; }}
   ws.onopen=()=>{{ conn.textContent="🟢 WebSocket connected"; send.disabled=false; status.textContent=""; }};
   ws.onmessage=(event)=>{{
     if(typeof event.data!=="string"){{status.textContent="⚠️ Received binary WebSocket frame";return;}}
     let d; try{{d=JSON.parse(event.data)}}catch(e){{status.textContent="⚠️ Invalid JSON from backend";return;}}
     if(d.type==="start"){{startAssistant();status.textContent="🔄 Agent is processing...";}}
     else if(d.type==="token"){{answer+=d.content||"";update();status.textContent="✍️ Receiving response...";}}
     else if(d.type==="answer"){{answer=d.content||"";update();}}
     else if(d.type==="metadata"){{tools=d.tools_used||[];sources=d.sources||[];prompt=d.prompt_version||"";backendModel=d.model_name||"";update();}}
     else if(d.type==="done"){{update();current=null;send.disabled=false;status.textContent="✅ Response completed. WebSocket remains open.";q.focus();}}
     else if(d.type==="error"){{if(!current)startAssistant();answer="❌ "+(d.message||"Unknown backend error.");update();current=null;send.disabled=false;status.textContent="";}}
   }};
   ws.onerror=()=>{{conn.textContent="🔴 WebSocket error";send.disabled=true;}};
   ws.onclose=(e)=>{{conn.textContent="🔴 WebSocket disconnected (code "+e.code+")";send.disabled=true;reconnect();}};
 }}
 function sendQuestion() {{
   const text=q.value.trim(); if(!text)return;
   if(!ws||ws.readyState!==WebSocket.OPEN){{status.textContent="⏳ WebSocket is not connected.";return;}}
   const row=document.createElement("div"); row.className="row user-row"; const bubble=document.createElement("div"); bubble.className="bubble user"; bubble.textContent=text; row.appendChild(bubble); msgs.appendChild(row);
   startAssistant(); status.textContent="📤 Sending question..."; send.disabled=true;
   ws.send(JSON.stringify({{query:text,model_type:MODEL}})); q.value=""; q.style.height="48px"; scroll();
 }}
 send.onclick=sendQuestion; q.onkeydown=e=>{{if(e.key==="Enter"&&!e.shiftKey){{e.preventDefault();sendQuestion();}}}}; q.oninput=()=>{{q.style.height="48px";q.style.height=Math.min(q.scrollHeight,150)+"px";}};
 connect();
}})();
</script></body></html>''', height=760, scrolling=True)


st.title("🤖 Agentic RAG Assistant")
st.caption("Base Qwen / Fine-tuned Qwen • Hybrid RAG • Wikipedia • DuckDuckGo • Browser WebSocket Streaming")

with st.sidebar:
    st.header("🤖 Agentic RAG")
    st.subheader("🧠 Model Selection")
    selected_model = st.radio("Choose model", ["finetuned", "base"], format_func=lambda x: "Fine-tuned Qwen (QLoRA)" if x == "finetuned" else "Base Qwen", index=0)
    if selected_model == "finetuned":
        st.info("Fine-tuned model\n\nQwen2.5-1.5B-Instruct + QLoRA adapter")
    else:
        st.info("Base model\n\nQwen2.5-1.5B-Instruct")
    st.divider()
    st.write("**Available Tools**")
    st.write("📄 Hybrid Document Search")
    st.write("🌐 Wikipedia")
    st.write("🔎 DuckDuckGo")
    st.write("🧠 Qwen Base / Fine-tuned")
    st.write("⚡ Browser WebSocket Streaming")
    st.write("📊 LangSmith Observability")
    st.divider()
    st.subheader("🔌 WebSocket")
    st.success("Browser WebSocket enabled")
    st.write("The actual chat connects directly from the browser to FastAPI /ws/agent.")
    st.code(f"ws://<your-pc-ip>:{BROWSER_WS_PORT}/ws/agent")
    st.caption("Open F12 → Network → WS. Ask a question and look for /ws/agent.")
    st.divider()
    st.subheader("📡 Flow")
    st.write("Browser → WebSocket → FastAPI → LangGraph → RAG/Tools → Qwen → WebSocket → Browser")

render_browser_chat(selected_model)
