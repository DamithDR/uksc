from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph

from experiments.lg.HuggingFaceLLM import HuggingFaceLLM
from experiments.lg.JudgmentState import JudgmentState


class Agent:
    def __init__(self, model, max_length, tools, system=""):
        self.system = system
        graph = StateGraph(JudgmentState)
        graph.add_node("llm", self.process_chunk)
        # graph.add_node("action", self.take_action)
        graph.add_node("predict_judgment", self.predict_judgment)
        graph.add_conditional_edges("llm", self.should_continue,
                                    {True: "llm", False: "predict_judgment"})
        # graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        graph.add_node("predict_judgment", END)
        self.graph = graph.compile()
        # self.tools = {t.name: t for t in tools}
        # self.model = model.bind_tools(tools)
        self.max_length = max_length



    def should_continue(self, state: JudgmentState):
        return state["current_chunk_idx"] < len(state["chunks"])

    # Process a single chunk of text (synchronous)
    def process_chunk(self, state: JudgmentState) -> JudgmentState:
        chunk = state["chunks"][state["current_chunk_idx"]]

        is_final = state['current_chunk_idx'] == len(state['chunks']) - 1

        prompt = PromptTemplate(
            input_variables=["chunk", "current_summary"],
            template="Given the following chunk of a legal judgment: '{chunk}', and the current summary of all previous chunks: "
                     "'{current_summary}', provide a concise summary of this chunk and then generate an overall summary for all the chunks given upto now."
            if is_final else "Given the final chunk of this legal judgment: '{chunk}', and the current summary of all previous chunks: "
                             "'{current_summary}', provide a concise summary of this last chunk and use that to provide a final summary of the whole judgment."
        )

        # prompt = PromptTemplate(
        #     input_variables=["chunk", "current_key_points"],
        #     template="Given the following chunk of a legal judgment: '{chunk}', and the current list of key points from all previous chunks: "
        #              "'{current_key_points}', identify the most important points in this chunk and provide them as a concise list. Then, append these points to the existing list to create an updated list of key points for all chunks processed so far."
        #     if not is_final else "Given the final chunk of this legal judgment: '{chunk}', and the current list of key points from all previous chunks: "
        #                          "'{current_key_points}', identify the most important points in this last chunk as a concise list. Then, append these points to the existing list to provide a final, comprehensive list of key points for the entire judgment."
        # )

        response = llm.generate(
            [prompt.format(chunk=chunk, current_summary=state["full_text_summary"] or "No summary yet.")], max_length)
        # response = llm.generate(
        #     [prompt.format(chunk=chunk, current_key_points=state["full_text_summary"] or "No summary yet.")], max_length)
        chunk_summary = response[0]["text"]
        state["chunks_processed"].append(chunk_summary)
        state["full_text_summary"] = chunk_summary  # Update running summary
        state["current_chunk_idx"] += 1  # Move to the next chunk
        return state

    def predict_judgment(self, state: JudgmentState) -> JudgmentState:
        prompt = PromptTemplate(
            input_variables=["summary"],
            # template="Based on the following summary of a legal judgment: '{summary}', predict the outcome as either "
            #          "'allow' or 'dismiss'. Provide a single-word answer.",
            template="""Assume you are a judge at the supreme court in United Kingdom. 
                        You will be provided UK supreme court appeal cases by the users and your duty is to understand the case background and output your decision label. 
                        Classify whether the provided appeal is allowed or dismissed, select one from following : [allow,dismiss].
                        Following is the summary of the judgment, please respond allow/dismiss, do not respond any explanation, other than allow/dismiss.
                        Summary : {summary}"""
        )
        response = llm.generate([prompt.format(summary=state["full_text_summary"])], max_length)
        prediction = response[0]["text"].lower()
        state["judgment_prediction"] = "allow" if prediction == "allow" else "dismiss"
        return state
