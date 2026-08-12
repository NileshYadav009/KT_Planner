from glossary import apply_glossary_corrections

t = "We use coffee for a sink event screaming between surfaces"
new, changed = apply_glossary_corrections(t, 0.0)
print('changed=', changed)
print('result=', new)
