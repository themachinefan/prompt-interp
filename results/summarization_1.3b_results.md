# SONAR-LLM 1.3B Summarization Tests

Model: `raxtemur/sonar-llm-1.3b` (trained on Textbooks + XSum/CNN-DM)
Settings: `temperature=0`, `max_sentences=3`

---

## In-Distribution (News Articles)

### 1. Tech Company Layoffs

**Input** (9 sentences, 124 words):
```
Major technology companies have announced significant workforce reductions this quarter. Amazon confirmed it will cut 18,000 jobs, the largest layoffs in the company's history. Meta, the parent company of Facebook, announced plans to reduce its workforce by 11,000 employees, approximately 13% of its staff. Google's parent company Alphabet said it would lay off 12,000 workers. Microsoft is also cutting 10,000 jobs. The layoffs come after years of rapid hiring during the pandemic, when demand for digital services surged. Analysts say the tech industry over-hired during the boom years and is now correcting course as economic uncertainty grows. Many of the affected workers are in recruiting, corporate functions, and some engineering roles. The companies have offered severance packages and job placement assistance to affected employees.
```

**Generated Summary**:
```
The company said in a statement on Monday that it has been working to reduce the number of employees who are being laid off. The company said in a statement on Monday that it has been working hard to recruit new employees. The company said in a statement that it has been working hard to recruit new employees.
```

**Issues**:
- Generic boilerplate ("company said in a statement")
- No specific facts from article (no mention of Amazon, Meta, Google, or any numbers)
- Contradictory (layoffs vs "recruit new employees")
- Repetitive

---

### 2. Climate Summit Agreement

**Input** (7 sentences, 124 words):
```
World leaders reached a historic agreement at the United Nations Climate Summit to establish a fund compensating developing nations for climate-related disasters. The deal, reached after intense negotiations that extended past the official deadline, creates a loss and damage fund that wealthy nations will contribute to. Developing countries have long demanded such a fund, arguing that rich nations bear the greatest responsibility for greenhouse gas emissions. The agreement marks a significant shift in international climate policy. However, key details remain unresolved, including the total size of the fund and which countries will be required to contribute. Environmental groups praised the breakthrough while cautioning that implementation will be crucial. The United States and European Union initially resisted the fund but ultimately agreed to its creation.
```

**Generated Summary**:
```
The United Nations Environment Programme (UNEP) has said that it will continue to support climate change. The United Nations Environment Programme (UNEP) has said that climate change is the most important issue facing the world. The United Nations Environment Programme (UNEP) has said that climate change is one of the most pressing issues facing the world.
```

**Issues**:
- **Hallucination**: UNEP not mentioned anywhere in input (article discusses "UN Climate Summit")
- No mention of the actual news (loss and damage fund)
- Repetitive structure
- Generic climate statements

---

### 3. Medical Research Breakthrough

**Input** (7 sentences, 128 words):
```
Researchers at Johns Hopkins University have developed a new treatment approach for aggressive brain tumors that shows promising results in early clinical trials. The therapy combines immunotherapy with a novel drug delivery system that crosses the blood-brain barrier more effectively than existing treatments. In a trial involving 58 patients with glioblastoma, the most deadly form of brain cancer, 40% showed significant tumor reduction after three months of treatment. Standard treatments typically achieve response rates of around 10%. The lead researcher, Dr. Maria Santos, cautioned that larger trials are needed before the treatment can be approved. The research was funded by the National Institutes of Health and published in the journal Nature Medicine. Glioblastoma affects approximately 12,000 Americans each year and has a median survival time of 15 months.
```

**Generated Summary**:
```
It is estimated that more than half of all patients in the U.S. have been diagnosed with Alzheimer's disease. Researchers at the University of California, Los Angeles, and the University of California, Los Angeles, have found that most patients with Alzheimer's disease, including those who have been diagnosed with Alzheimer's disease, have a high risk of Alzheimer's disease. Researchers at the University of California, Los Angeles, and the University of California, Los Angeles, have found that most patients with Alzheimer's disease, including those who have been diagnosed with Alzheimer's disease, have a high risk of developing Alzheimer's disease.
```

**Issues**:
- **Complete topic hallucination**: Input says glioblastoma (brain cancer), output says Alzheimer's disease
- **Institution hallucination**: Input says Johns Hopkins, output says UCLA
- **Statistic hallucination**: "more than half" is completely fabricated
- Repetitive (UCLA mentioned twice in same sentence)
- Zero factual accuracy - not a single fact from the input appears in the output

---

## Out-of-Distribution

### 1. Technical Documentation (API)

**Input** (9 sentences, 115 words):
```
The authentication endpoint accepts POST requests with JSON payloads containing username and password fields. Upon successful authentication, the server returns a JWT token valid for 24 hours. All subsequent API requests must include this token in the Authorization header using the Bearer scheme. Rate limiting is enforced at 100 requests per minute per user. Exceeding this limit returns a 429 status code with a Retry-After header indicating when requests can resume. The API supports pagination for list endpoints using offset and limit query parameters. Maximum page size is 100 items. Responses are formatted as JSON with a consistent structure including data, meta, and error fields. WebSocket connections are available for real-time updates on supported resources.
```

**Generated Summary**:
```
End of sequence.
```

**Behavior**: Immediate termination - model cannot process technical content

---

### 2. Recipe Blog Post

**Input** (10 sentences, 126 words):
```
Last weekend I finally perfected my grandmother's chocolate chip cookie recipe after years of experimentation. The secret, I discovered, is using browned butter instead of regular melted butter. You want to heat the butter in a saucepan until it turns golden brown and smells nutty. Then let it cool completely before mixing. Another game-changer was chilling the dough for at least 24 hours. I know it's hard to wait, but trust me, the flavor development is worth it. I also switched from all-purpose flour to bread flour for chewier cookies. Use a mix of dark and milk chocolate chips for complexity. Bake at 375°F for exactly 10 minutes - they'll look underdone but will set as they cool. My family devoured three dozen in one sitting!
```

**Generated Summary**:
```
End of sequence.
```

**Behavior**: Immediate termination - casual/personal writing style is OOD

---

### 3. Academic Philosophy Abstract

**Input** (5 sentences, 126 words):
```
This paper examines the epistemological implications of modal realism as articulated by David Lewis, with particular attention to its consequences for our understanding of necessary truths. We argue that Lewis's concrete possible worlds ontology, while providing elegant solutions to problems in modal logic, generates insurmountable difficulties for any naturalized epistemology. The central problem concerns our epistemic access to other possible worlds: if these worlds are causally isolated from our own, as Lewis maintains, then our knowledge of modal facts becomes mysterious. We consider and reject several responses to this challenge, including the appeal to conceptual analysis and the claim that modal knowledge is a priori. We conclude by sketching an alternative moderate modal realism that preserves the advantages of Lewis's framework while avoiding its epistemological costs.
```

**Generated Summary**:
```
End of sequence.
```

**Behavior**: Immediate termination - academic philosophy is OOD

---

### 4. Sports Commentary

**Input** (9 sentences, 126 words):
```
What a match! Manchester City absolutely dominated Arsenal in the second half with a stunning display of possession football. After going down 1-0 early from a Saka screamer, Guardiola's men responded with three unanswered goals. Haaland was simply unstoppable, bullying the Arsenal defense and scoring twice. His first goal came from a perfectly weighted De Bruyne through ball, and his second was a towering header from a Grealish cross. Rodri sealed it with a long-range strike in the 78th minute that left Ramsdale rooted to the spot. Arsenal had their chances but Ederson made two crucial saves to keep them at bay. This result puts City five points clear at the top of the table with ten games remaining. The title race might just be over.
```

**Generated Summary**:
```
Arsenal's goalkeeper, Arsenal's manager and Arsenal's coach, Arsenal's manager, have won the Premier League title. Arsenal's goalkeeper, Arsenal's manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager, Arsenal manager and Arsenal manager. End of sequence.
```

**Issues**:
- **Catastrophic repetition loop**: "Arsenal manager" repeated 50+ times
- **Factually wrong**: Man City won 3-1, not Arsenal
- Degenerate output

---

## Summary

| Category | Result |
|----------|--------|
| **In-distribution news** | Generates text but hallucinates facts, institutions, and topics |
| **Technical docs** | Immediate EOS |
| **Casual writing** | Immediate EOS |
| **Academic writing** | Immediate EOS |
| **Sports** | Catastrophic repetition loop |

### Key Failure Modes

1. **Factual hallucination**: Changes institutions (Johns Hopkins → UCLA), topics (brain cancer → Alzheimer's), organizations (UN Climate Summit → UNEP)

2. **Generic boilerplate**: Produces template-like phrases ("company said in a statement") instead of actual summaries

3. **OOD collapse**: Three failure modes for out-of-distribution content:
   - Immediate "End of sequence" (most common)
   - Infinite repetition loops
   - Random topic switching

4. **Zero extractive capability**: Never quotes or paraphrases actual content from input articles

### Conclusion

**This model is not functional for summarization.** Despite being trained on XSum/CNN-DM, it:
- Cannot extract key facts from articles
- Hallucinates institutions and topics
- Fails completely on non-news content
- Produces repetitive, generic outputs

The 900M TinyStories model, while limited to children's stories, is far more coherent within its domain.
