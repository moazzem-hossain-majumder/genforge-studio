"""
Lab 9: Job application tools — Resume Polisher, Cover Letter Generator,
Career Advisor.

Original lab used watsonx.ai (meta-llama/llama-3-2-11b-vision-instruct).
Replaced with the shared local instruct LLM and the same prompts.
"""

import gradio as gr

from core.models import instruct_chat


def polish_resume(position_name, resume_content, polish_prompt=""):
    if polish_prompt and polish_prompt.strip():
        prompt = (
            f"Given the resume content: '{resume_content}', polish it based on "
            f"the following instructions: {polish_prompt} for the {position_name} position."
        )
    else:
        prompt = (
            f"Suggest improvements for the following resume content: "
            f"'{resume_content}' to better align with the requirements and "
            f"expectations of the {position_name} position."
        )
    return instruct_chat(
        [{"role": "user", "content": prompt}], max_new_tokens=400, temperature=0.7
    )


def generate_cover_letter(company_name, position_name, job_description, resume_content):
    prompt = (
        f"Generate a customized cover letter using the company name: {company_name}, "
        f"the position applied for: {position_name}, the job description: "
        f"{job_description}, and the resume: {resume_content}."
    )
    return instruct_chat(
        [{"role": "user", "content": prompt}], max_new_tokens=450, temperature=0.7
    )


def generate_career_advice(position_applied, job_description, resume_content):
    prompt = (
        f"Considering the position applied for: {position_applied}, the job "
        f"description: {job_description}, and the resume provided: {resume_content}, "
        f"identify areas for enhancement in the resume to better match the job "
        f"requirements, and give advice on career development."
    )
    return instruct_chat(
        [{"role": "user", "content": prompt}], max_new_tokens=500, temperature=0.7
    )


def build_tab():
    with gr.Tab("💼 Career Tools"):
        gr.Markdown("### Job Application Tools")

        with gr.Tab("Resume Polisher"):
            pos = gr.Textbox(label="Position Name")
            resume = gr.Textbox(label="Resume Content", lines=15)
            instr = gr.Textbox(label="Polish Instruction (optional)", lines=2)
            btn = gr.Button("Polish")
            out = gr.Textbox(label="Polished Content", lines=15)
            btn.click(polish_resume, inputs=[pos, resume, instr], outputs=out)

        with gr.Tab("Cover Letter Generator"):
            company = gr.Textbox(label="Company Name")
            pos2 = gr.Textbox(label="Position Name")
            jd = gr.Textbox(label="Job Description", lines=8)
            resume2 = gr.Textbox(label="Resume Content", lines=8)
            btn2 = gr.Button("Generate Cover Letter")
            out2 = gr.Textbox(label="Customized Cover Letter", lines=15)
            btn2.click(generate_cover_letter, inputs=[company, pos2, jd, resume2], outputs=out2)

        with gr.Tab("Career Advisor"):
            pos3 = gr.Textbox(label="Position Applied For")
            jd3 = gr.Textbox(label="Job Description", lines=8)
            resume3 = gr.Textbox(label="Your Resume Content", lines=8)
            btn3 = gr.Button("Get Advice")
            out3 = gr.Textbox(label="Advice", lines=15)
            btn3.click(generate_career_advice, inputs=[pos3, jd3, resume3], outputs=out3)
