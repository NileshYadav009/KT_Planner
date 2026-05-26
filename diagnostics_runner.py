from ai import build_section_paragraphs
transcript = (
    "The system name is cloud native order processing platform. "
    "This system handles order intake, validation, payment orchestration and fulfillment triggers. "
    "It is used by B2C users through the web app, B2B partners through APIs, and internal finance and support teams. "
    "This system is most critical during business hours and peak sales events."
)
def run():
    try:
        paras = build_section_paragraphs(transcript)
        print('paragraphs:', paras)
        print('keys:', list(paras.keys()))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('ERROR build_section_paragraphs', e)

    # Directly call create_semantic_mapper and inspect internals
    try:
        from ai import SCHEMA
        from enterprise_semantic_mapper import create_semantic_mapper
        sentences = [(f'sent_{i}', s.strip()) for i, s in enumerate(transcript.split('.')) if s.strip()]
        mapper = create_semantic_mapper(SCHEMA)
        result = mapper.process_transcript(sentences)
        print('\nmapper.process_transcript result keys:', list(result.keys()))
        print('  paragraphs keys:', list(result.get('paragraphs', {}).keys()))
        print('  assignments keys:', list(result.get('assignments', {}).keys()))
        print('  unclassified count:', len(result.get('unclassified', [])))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('ERROR mapper', e)

if __name__ == '__main__':
    run()
