<!-- title: Programming Idioms -->
<!-- date: 2021-08-21-->
<!-- syntax_highlighting: on -->

Programming, in a modern sense, is the transfer of mathematically well-defined structure and logic between a highly advanced biological computer with bounded computing rates, unreliable precision, but much common sense, and a fast mechanical computer with growing processing rates, highly reliable precision, but with absolute zero common sense. In other words, it's a smart, slow computer trying to dumb down simple instructions for a fast, dumb computer, because there's value in making that communication happen.

For there to even be a communication channel, the human has to make the compromise, and the compromise is that they enter an environment in which the expression of concepts and algorithms is extremely limiting, error prone, and slow.

Because of this compromise, great care needs to be taken towards managing the workflow and one needs to minimize the amount of time spent debugging. Therefore, programing is not actually about algorithms or theory; those are details that can be solved by looking up how others before you solved your problem, consulting a resource, or just thinking anyways. It's really about predicting, acknowledging, paying attention to, and finding your limitations and weaknesses so that you can prevent yourself from making mistakes, and so you can make smart tradeoffs that make your code humanistic. All of this is just within the best interest of efficiency.

Here are some rules that I've learned from others or have found on my own that help me make less mistakes.

#### Try not to be lazy or impatient.
If you are lazy or impatient, then you increase the probability of causing bugs, and debugging will reduce your productivity. You can also be courteous and do what you can to simply minimize the next programmer's misery in the case that you make a bug.

* **Avoid misleading or vague variable names.** Examples: Don't use "val" for value or "res" for "result" because these do not really mean anything specific.

* **Keep variable names short**. More specifically, most variable names are a concatenation of one to three descriptive words that describe what the variable is supposed to reference. Using more than three words in name makes the code clunky, and usually there's a way to keep it short. If you do find your variable name needs to be long, either your work project is massive and that's your tradeoff, or there's something off with the implementation.

* **Try not to write long comments**. This usually happens because what someone is doing is a little more complicated than the average block of code. However, clean code always has really short comments. 

* **Avoid the opposite: short, vague or meaningless comments.** I have found that this is worse than no comments. Spending 5 - 20 seconds thinking of an informative, brief, truthful comment is better than causing a butterfly effect of a bug that then causes minutes to hours of debugging later. 

* **Keep functions short**. If you do this and split up your procedures nicely, you can detect bugs easily since you can test smaller procedures first. The exception to this is when there are a lot of different cases for a function's behavior and breaking it down would actually introduce too much black-boxing and disorganization. I've noticed parsers tend to be very, very long functions, and I imagine it's for this reason. 

* **Avoid using a single letter variable name**. For example, if you're doing a for loop and the loop is over indices, try "ind" instead of "i" or "j". An exception is something like 
```python
# In Python, f is a commonly used variable name for a file opened via a context manager. 
with open(file) as f:
    res = f.read()  # Not the best variable name...
```

* **Don't include the type of a variable in its name** (e.g "angle_int" for an angle between 0 and 360). It will cause confusion in a duck typing language and it's usually unnecessary.

* **Use one-liners conservatively**. I have seen what happens whent they're abused; they can become the source of a silly bug, especially if for instance a one liner makes two calls to the same function; a vertical traceback will not help you much then. Exceptions are simple ternary operators, since these are pretty handy.

* **If the implementation of an idea feels wrong, do everything you can to resist saying "I'll come back to it."** This is because when this happens, you're already in the moment and have the focus, so it's best to get it done with then. Additionally, if you do it once, you likely do it multiple times, and then you'll build up a stack of "I'll come back to it"'s that you won't come back to, and then even if you do come to these things, you'll have to spend time remembering what it was that you were doing. Of course, some ideas do need to sit around in the subconscious, but that's a process that is supposed to happen before an implementation.

* When you finish writing a block of code, you should read over it a once or twice and *then* test it, no matter how simple the function is, even if it is literally adding two numbers. This is so you can be lazy later, so that when a bug happens (and a bug will happen) you minimize the probability that you will have to check for many places to find and fix the bug. That is, it decreases the likelihood that your bug is in a quantum superposition of two or more functions, and so you may save yourself a lot of time and effort by doing this.

This is actually the most important rule.

#### Be as lazy as you can be.
In other words: Don't make yourself work harder than you have to because it deters your focus from the actual code.

* **Resist the temptation to manually format your code** (e.g., inserting a few spaces, putting a comment on a different line, resorting your import statements). This is a waste a time, a distraction from your main focus, and the code conventions will become visbily out of sync. If you're like me and you've caught yourself doing it anyway, the way you solve this forever is by using a code formatter.


* **Don't force yourself to remember different keybindings for different GUIs and editors**. I've found that this will lead to silly errors and brief moments of "How do I do it in this one again?". The way you solve this problem is by learning a *sensible and widely* used editor like Vim (which I recommend, but ocassionally I regret this because its clipboard functionality drives me nuts), and configure your applications to use it so that you only ever have to remember one set of keybindings for everything. 

* **Do not use your default terminal settings**. Default terminal settings are a gigantic time waster (and that's because they're supposed to be upgraded). With default terminal settings, you'll find yourself slowly pressing your backspace 20 times to the left to get to the beginning of a command, because you forgot "sudo", and 20 times again to the right to get back to the end, because you forgot an extra command flag. You will truly change your life by putting good keybindings (like Vim) on your terminal.

* **Write down code snippets**. Pick a dead simple code snippet application, create folders for different programming languages/applications, and put your frequently used commands in there. This will prevent you from wasting time on tracking down that-one-really-helpful-SO-post-that-I-now-cant-find, and you will thank your past self for saving your future self time.

* **Don't ever worry about possible performance improvements, unless you have a reason to**. You should only go so far as to focus on not writing obviously silly code that's doing more work than it needs to.

* **Avoid resorting to shared memory, multiprocessing/threading, and low-level programming** whenever you can. Especially don't do it for speed, because it can sometimes be overkill and a nightmare. I learned this the hard way!


Another rule is: 
#### Try to make sure you aren't fooling yourself.
To really be safe, you must assume that, even if you are currently bug free, *your code will fail someday*. 

* **You should always assume your code and your logic is wrong in some way.** There is never bug free code. Your code is a program waiting to crash and fail because some fruit company decided to use a different processing chip.

* **Never fool yourself into thinking you are done improving your skills**. There is always more to learn and there always will be.

* **Always assume that someone can show you things you've never seen before**. The main thing you can learn from others is how they think about and solve a problem.

* **Find code written by other people and study it**. I have learned a lot by doing this when I was just actually trying to figure out how something was working. And this is really easy these days, just find the repository for a high quality library you frequently use. 


