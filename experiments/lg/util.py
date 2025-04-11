from langchain_core.prompts import PromptTemplate

from experiments.lg.JudgmentState import JudgmentState


def process_chunk(llm, state: JudgmentState, max_length: int) -> JudgmentState:
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
        [prompt.format(chunk=chunk, current_summary=state["full_text_summary"])], max_length)
    print(response)
    # response = llm.generate(
    #     [prompt.format(chunk=chunk, current_key_points=state["full_text_summary"] or "No summary yet.")], max_length)
    chunk_summary = response[0]["text"]
    state["chunks_processed"].append(chunk_summary)
    print(chunk_summary)
    state["full_text_summary"] = chunk_summary  # Update running summary
    state["current_chunk_idx"] += 1  # Move to the next chunk
    return state


def predict_judgment(llm, state: JudgmentState, max_length: int) -> JudgmentState:
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
