import sys, json, time
sys.path.insert(0, '.')
from context_mapper import ContextMappingPipeline

schema = json.load(open('kt_schema_new.json'))['sections']
model = ContextMappingPipeline(schema, enterprise_mapper_enabled=False)
text = 'We use kubernetes to deploy the application and helm to manage releases. Monitoring uses Prometheus and Grafana. If the pod fails, we restart with kubectl rollout restart deployment.'
segments = [{'text': text, 'start': 0.0, 'end': 30.0, 'avg_logprob': -0.1}]

start = time.time()
kt = model.process('debug', text, segments)
print('process time', time.time() - start)
print('overall coverage', kt.overall_coverage_percent)
print('status counts', {sec_id: cov.status for sec_id, cov in kt.coverage.items() if cov.status != 'missing'})
print('section content count', len(kt.section_content))
print('unassigned count', len(kt.unassigned_sentences))
for i, cs in enumerate(kt.classified_sentences):
    print(i, repr(cs.sentence.text), '->', cs.primary_classification.section_id if cs.primary_classification else None, 'conf', cs.primary_classification.confidence if cs.primary_classification else None, 'unassigned', cs.is_unassigned)
