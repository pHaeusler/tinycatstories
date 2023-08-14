# tinycatstories

An experiment in aligning [tinystories](https://arxiv.org/abs/2305.07759) to only tell stories about cats.

We use a modified [REINFORCE algorithm](http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf) with [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) to RL the tinystories model without any additional data.

This is like a poor mans [PPO](https://huggingface.co/blog/deep-rl-ppo).

The reward model is an embedding similarity to the text "cat".

[Blog Post](https://philliphaeusler.com/posts/aligning_tinystories/)

---

Train with: `python reinforce.py`

Test the stories with: `python sample.py`

**Example Story**

```
Once upon a time there was a cat. The cat wanted to find food, so he set out to find some.

He walked and walked until he found a big basket of food. It smelled delicious and he was excited! So he started to run towards the basket of food.

He reached it and tried to grab some food. But the cat was not very careful and he hurt his paw. He yipped and yawed and he looked very sad!

Just then a kind dog came by and saw the cat. He shouted, “What are you doing in my basket?”

“Oh, I was just trying to eat some food!” said the cat.

“Well, you don’t have to be so hungry all the time,” said the kind dog.

The cat felt so happy and he knew that being careful was the best way to enjoy the food.

The end.

```
