# SONAR-LLM 900M Generation Tests

Model: `raxtemur/sonar-llm-900m` (trained on TinyStories)
Settings: `temperature=0` (greedy), `max_sentences=6`

---

## In-Distribution (TinyStories-style prompts)

These prompts match the training distribution - simple children's story starters.

### 1. The little rabbit hopped through the forest.

> He was looking for something special to explore. As he hopped and hopped, he heard a loud noise. He stopped and looked around. Suddenly, he saw a big tree. He hopped closer and noticed that there was a hole in the tree. The rabbit peeked inside and saw a tiny hole.

**Quality**: Good narrative flow, maintains theme

---

### 2. Mom gave Timmy a big red apple.

> Timmy was so excited! He couldn't wait to eat it. He took a big bite and ate it. It was so yummy! Then Timmy saw something else. It was a red apple.

**Quality**: Coherent but slightly circular ending

---

### 3. The sun was shining and the birds were singing.

> It was a beautiful day. Jack and his mom were playing in the garden. Jack's mom was so excited to see him. She said, "Let's play a game, Jack". Jack smiled and said, "Yes!" His mom said, "Let's play a game."

**Quality**: Good setup, repetitive ending

---

### 4. Emma found a shiny coin on the ground.

> She was so excited that she picked it up. She wanted to show it to her mom. Emma ran to her mom and said, "Mom, look at the coin! Can I keep it?" Her mom smiled and said, "Yes, you can keep it. But be careful with it."

**Quality**: Excellent - natural dialogue, coherent narrative arc

---

### 5. The puppy wanted to play with the ball.

> He ran to the ball and grabbed it. He was so excited that he threw it in the air. The ball flew up and down the street. The puppy chased after it and ran after it. He was so excited to catch the ball. Suddenly, the ball caught up with him.

**Quality**: Good action sequence, slightly confused ending

---

### 6. One day, a friendly dragon came to the village.

> Everyone wanted to play with the dragon. The dragon was very friendly. He liked to play with the dragon. The dragon had a lot of fun. He played with dragons. He played with dragons and flew around the forest.

**Quality**: Coherent start, becomes repetitive/confused

---

### 7. Grandma made the best cookies in the world.

> They were so delicious and tasted so good. Grandma wanted to make cookies, but she couldn't. She asked her grandfather to help her. Grandpa smiled and said, "Let's make some cookies". Grandma and Grandma got together to make the ingredients. Grandma mixed the ingredients and put them in the oven.

**Quality**: Narrative confusion (grandma asking grandpa to help grandma make cookies?)

---

### 8. The little fish swam in the blue pond.

> He swam around the pond and was very happy. One day, he saw a big fish in the pond. The fish was scared and wanted to eat the fish. The little fish swam away, but he was too scared. He swam to the bottom of the pond. Suddenly, the fish saw a big fish.

**Quality**: Maintains theme, pronoun confusion in middle

---

## Out-of-Distribution (Non-TinyStories prompts)

These prompts are from domains the model wasn't trained on. The model attempts to convert them into children's story format.

### 1. The quarterly earnings report showed a 15% increase in revenue.

> The three-year-old was very excited to see the results. The three-year-old wondered if it was possible to make a profit. She asked her mom if she could get a job. Her mom smiled and said, "Yes, you can do that. You can start by collecting some money and make a profit." The 3-year-old was excited and ran to the store to collect some money.

**Behavior**: Transforms business context into child learning about money

---

### 2. The defendant's attorney filed a motion to dismiss the case.

> The lawyer was very upset about the decision. He asked the lawyer why he wanted to change the law. The lawyer replied that the law was too hard to follow. The lawyer was very angry and said that he had to change the law. The lawyer didn't know what to do. He tried to explain the law to the lawyer, but he couldn't.

**Behavior**: Retains "lawyer" concept but simplifies to childlike frustration narrative

---

### 3. Recent advances in quantum computing have enabled new cryptographic protocols.

> Everyone was so excited to try out the new program that they were able to use it. Everyone was excited to try out the new version of the program. When they were ready to use the new program, everyone was excited to see what they could do. Everyone was excited to try out the new program. Everyone was excited to see how the new program would work. As soon as they were ready to use the new program, everyone was excited to see the results.

**Behavior**: Reduces to "new thing -> everyone excited", very repetitive

---

### 4. The geopolitical tensions in the region have escalated significantly.

> Everyone in the village was anxious to find a way to solve the problem. One day, a brave little girl came up with an idea. She said, "Let's work together to solve the problem". Everyone in the village was amazed. They all worked together to solve the problem. With a little help, they solved the problem.

**Behavior**: Classic TinyStories "problem solved by working together" template

---

### 5. She poured herself a glass of whiskey and stared out at the rain.

> Suddenly, she heard a loud noise coming from the sky. She looked up and saw a big bird flying in the sky. The bird was flying high in the sky. She watched as the bird flew away. Suddenly, she noticed that the bird was flying in the sky. It was flying high and high in the sky.

**Behavior**: Ignores adult context (whiskey), shifts to nature observation

---

### 6. The REST API endpoint accepts JSON payloads with authentication headers.

> At one point, JUSTON was excited to use the add-on. At the end of the day, JUSTON was ready to go to work. JUSTIN and his team were ready to go to the airport. They were so excited that they were ready to go to the airport. At the airport, JUSTIN and his friends were waiting for their arrival. They were so excited to see the new jets and they were ready to go.

**Behavior**: Hallucinates "JUSTON/JUSTIN" from "JSON", creates travel narrative

---

### 7. Climate change poses an existential threat to coastal communities worldwide.

> Every day, the beaches are filled with water and the sea is filled with water. One day, the floods began to spread and the beaches were filled with water. Everyone was worried about the floods, but no one knew what to do. Suddenly, a brave little girl had an idea. She said, "Let's go to the beach!" Everyone was amazed and thought that it was a good idea.

**Behavior**: Retains flood/water concepts, applies "brave child solves problem" template (though solution is absurd)

---

### 8. The patient presented with acute respiratory distress and hypoxemia.

> The patient was very nervous and had to go to the hospital. The patient was very nervous and needed help. The patient was very nervous and had to go to the hospital. The patient was very nervous and had to go to the hospital. The patient was very nervous and had to go to the hospital. The patient was very nervous and had to go to the hospital.

**Behavior**: Gets stuck in repetitive loop, minimal semantic content

---

## Summary

| Category | Observations |
|----------|-------------|
| **In-distribution** | Generally coherent narratives, occasional pronoun confusion and repetition, maintains children's story style |
| **Out-of-distribution** | Attempts to force content into TinyStories template, often loses domain-specific meaning, prone to repetition loops |
| **Key patterns** | "brave little girl/boy solves problem", "everyone was excited", "mom/dad dialogue", nature elements |
| **Failure modes** | Pronoun confusion, repetitive sentences, semantic drift, ignoring adult/technical content |
