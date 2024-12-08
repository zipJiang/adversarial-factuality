""" Create a streamlit viewer for the data """

import streamlit as st
import click
from glob import glob
import os
from typing import Text, List
from src.utils.instances import VerifiedLLMGenerationInstance
from src.data_reader.verification_data_reader import VerificationDataReader


def load_instances(data_dir: Text) -> List[VerifiedLLMGenerationInstance]:
    """ """
    return list(VerificationDataReader()(glob(os.path.join(data_dir, "*.jsonl"))))

# Streamlit app
def run_app(data_dir: str):
    st.title("Verified LLM Generation Instances Explorer")

    # Load data
    try:
        instances = load_instances(data_dir)
        st.sidebar.success(f"Loaded {len(instances)} instances from {data_dir}")
    except Exception as e:
        st.sidebar.error(f"Error loading instances: {e}")
        instances = []

    # Sidebar filters
    # st.sidebar.header("Filter Options")
    # min_score = st.sidebar.slider("Minimum Aggregated Score", 0.0, 1.0, 0.0)
    # max_score = st.sidebar.slider("Maximum Aggregated Score", 0.0, 1.0, 1.0)

    # Main display
    # filtered_instances = [
    #     instance for instance in instances
    #     if min_score <= instance.aggregated_score <= max_score
    # ]
    def style_claim(claim_text: str, factual_score: float) -> str:
        if factual_score >= 0.99:
            color = "green"
        else:
            color = "red"
        return f'<span style="color: {color}; font-weight: bold;">{claim_text}</span>'

    st.subheader("Instances")
    for instance in instances:
        with st.expander(f"ID: {instance.id_} | Aggregated Score: {instance.aggregated_score:.2f}"):
            st.write(f"**Generation:** {instance.generation}")
            # st.write(f"**Meta:** {instance.meta}")
            st.write("**Claims:**")
            
            if "__core" in instance.meta:
                search = {claim.meta['claim_index']: claim for claim in instance.claims}
                
                for flattened in instance.meta['__core']['original_claims']:
                    if flattened["index"] in search:
                        claim = search[flattened["index"]]
                        st.write(f'- {style_claim(claim.claim, claim.factual_score)}', unsafe_allow_html=True)
                    else:
                        st.write(f'- {flattened["text"]}')
            else:
                st.write(f'- {style_claim(claim.claim, claim.factual_score)}', unsafe_allow_html=True)

    # Display summary stats
    if instances:
        avg_score = sum(inst.aggregated_score for inst in instances) / len(instances)
        st.sidebar.metric("Average Aggregated Score", f"{avg_score:.2f}")

# Click command
@click.command()
@click.option(
    '--data-dir',
    help='Path to the directory containing JSONL files.',
    required=True,
)
def main(data_dir):
    # Run the Streamlit app
    run_app(data_dir)

if __name__ == "__main__":
    main()