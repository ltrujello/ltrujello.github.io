<!-- title: GNU Readline and History Library -->
<!-- syntax_highlighting: on -->
<!-- preview_img: /cs/gnu_readline/dependency.png -->
<!-- date: 2021-11-02 -->

In this post, we'll show how to use the GNU Readline and History C libraries to design a REPL (read-evaluate-print-loop) program and to understand how terminal shells that we use every day obtain their nice features. Using these libraries we'll create a REPL shell with the following features:

* Inline editing and text traversal with left and right keys
* Command history via up and down arrow keys
* A smart history file on disk (smart in that it does not store empty commands, consecutive, duplicate commands, and it periodically flushes it contents)

These are the features exhibited by most command shells like Bash, Zsh, iPython, etc. 

# Background 
The GNU Readline and History libraries are C libraries that allows command line REPL programs to have nice inline editing and history features. They do a lot more, but it basically takes care of serving editor-like terminal input for the user and dealing with their command history.

GNU Readline includes the History library, so when we talk about GNU Readline, we're talking about both.

All kinds of programs use GNU Readline behind the scenes, including Bash and MySQL to name a few. Because these features dramatically speed up our workflow and are built into our muscle memory, and since not a lot of people know about Readline (I also didn't for a long time), a not accurate but funny way to view GNU Readline is with the following xkcd comic.

<img src="/cs/gnu_readline/dependency.png" style="margin: 0 auto; display: block; width: 30%;"/> 

The GNU Readline library does not provide all of our desired features listed above out of the box, but it does provide the tools to implement these features. I found the [documentation](https://tiswww.case.edu/php/chet/readline/rltop.html#Documentation) to be a bit terse, or at least I had many "in situation X what does Y do when Z happens" type of questions that the documentation did not answer that. I answered such questions myself by experimenting, looking at the headers, and thinking of the ways in which the program could crash. Since I imagine others would want to implement these features in their own programs, hopefully what I've learned can save others time. 

# A note about MacOS
If you plan on using the Readline library and you're developing on MacOS, make sure you're actually using GNU Readline and not some other imposter header files. 

In my experience my version of Readline was missing the `append_history` function, which is extremely important. It also would not record history unless I had something specific at the top of my history file. I eventually found I wasn't using GNU Readline. I'm not sure why this happened or who decided to create a fake readline library in MacOS's include files. Since this not how GNU Readline behaves, if you're having trouble getting Readline to work on a Mac this might be why. Read about the compilation below to see how to get it to work.

# Basic example
We'll first show how to use Readline to create a very simple REPL shell. This will allow you to see how Readline can very nicely deliver the text-editor like experience at a command prompt.

```C
/* first_ex.c */
#include <stdlib.h>
#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

int main(){
    char *input;
    while (1) {
        /* Display prompt and read input */
        input = readline("prompt> ");

        /* Check for EOF */
        if (!input)
            break;

        /* Add input to readline history */
        add_history(input);

        /* Print the input with an exclamation point. */
        printf("%s!\n", input);

        /* Free buffer that was allocated by readline */
        free(input);
    }
    return 0;
}
```
If you run this, you will get a shell that accepts input and screams back at you with the input you gave it. You can navigate over characters with your left and right arrow keys. Here, `add_history(input)` keeps track of a "history list" and we can use our up and down arrow keys to access previously entered input.

# Compilation 
To compile this you must use the `-lreadline` flag at the end of the command. So something like 

```bash
gcc -o first_ex firest_ex.c -lreadline
``` 
will do. 

If your program can't find Readline, that means it could not find the file `lreadline.so` in directories like `/usr/lib` or `/usr/local/lib`. This would particularly odd, and you might want to update your GNU coreutils or you can just simply get a copy of the source of the Readline library (check how to do it on your system, e.g. for a Debian-based Linux, `sudo apt install libreadline-dev` worked for me). You can also try finding this file, and explicitly giving the path to your Readline library via `-L` flag. Then 

```bash
gcc -o first_ex firest_ex.c -Lpath/to/some/dir/ -lreadline 
```
will compile the program. 

# Manually Installing
If you're forced to manually install, which might happen if you have MacOs, run these commands (note that you should replace the ftp url with the latest .tar.gz of GNU readline, which you can find [here](https://tiswww.case.edu/php/chet/readline/rltop.html)).
```bash
curl -L -o readline-8.1.tar.gz ftp://ftp.gnu.org/gnu/readline/readline-8.1.tar.gz 
tar -xvzf readline-8.1.tar.gz 
cd readline-8.1
./configure
make
make install
```
The output of `make install` will tell you where it installed the library on your machine. I did this on a MacOS and that was `/usr/local/include/readline` for me. You can now supply this directory to the `-L` command argument in the compilation command.


# Creating and using a history file
Our first program isn't very good, because if we close the program we lose our command history. To remedy this, we need to create a history file that tracks our commands.

GNU Readline provides some functions to assist with this, the three most useful being the following.

* `int read_history(const char *filename)` will receive a *full path* to the history file and add the contents of the history file to the program's history list.

* `int add_history(const char *input)` will **copy the pointer** `input` and add it to the history list. This is what allows you to get your history via up and down arrow keys.

* `int append_history(int nelements, const char *filename)` will append the most `nelements`-recent commands to the hisory file with full path `filename`.

Both of these return 0 if successful, otherwise some error number. If `filename` is a `NULL` pointer, it defaults to the `~/.history` file (which typically wouldn't exist). 

To make things simple, we will suppose our user's home directory is `/home/michael_scott` (Yes, it turns out he runs Linux) and the name of the file is `.myhistory`. **We will also suppose the file is created already.** The code to update a history file is then as follows.

```C
/* second_ex.c */
#include <stdlib.h>
#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>

int main(){
    char *input;
    /* Add history file content to our current history list */
    read_history("/home/michael_scott/.my_history");
    while (1) {
        /* Display prompt and read input */
        input = readline("prompt> ");

        /* Check for EOF */
        if (!input)
            break;

        /* Add input to history list */
        add_history(input);

        /* Add input to the history, but only if it is not blank content */
        if (*input)
            append_history(1, "/home/michael_scott/.my_history");

        /* Print the input with an exclamation point. */
        printf("%s!\n", input);

        /* Free buffer that was allocated by readline */
        free(input);
    }
    return 0;
}
```
Assuming `/home/michael_scott/.myhistory` already exists, this program will record and save command history every time input is entered into the program. 

# A better version that still saves history
Our last program is better since it saves history, but there are a number of problems. We address three important issues.

* The program will not work across all machines, since obviously our path to the history file `/home/michael_scott/.myhistory` is hardcoded. 

To address this concern, we can write the following function to calculate the full path to the history file. This should work for Unix-like machines that have a HOME variable defined.

```C
/* Must have #include <pwd.h> */

/* Get the full path of the history filename */
char *get_history_filename(void){
    /* First get the home directory of the user */
    char *homedir;
    if ((homedir = getenv("HOME")) == NULL)
        homedir = getpwuid(getuid())->pw_dir;

    /* Create the full path */
    size_t needed = snprintf(NULL, 0, "%s/.my_history", homedir);
    char *filename = malloc(needed + 1);
    if (filename == NULL)
        printf("Cannot allocate char pointer for history filename");
    return filename;
}
```

* Another concern is that we assumed the history file already existed. If the history file doesn't exist, the program will crash or silently fail at `read_history`. The following code can address this by creating the file if it doesn't exist. 

```C
/* Create the history file if it doesn't exist */
void create_history_file(const char *history_filename){
    if (access(filename, F_OK) != 0) {
        FILE *hist_file = fopen(history_filename, "w");
        if (hist_file == NULL) {
            fprintf(stderr, "Can't open file %s", history_filename);
            exit(1);
        }
        fclose(hist_file);
    }
}
```

* Another concern is that the history file is destined to grow arbitrarily large. We need to periodically truncate it somehow.

To address this third issue, GNU Readline provides the following function.

* `int history_truncate_file(const char *filename, int nlines)` will truncate the file, leaving only the last `nlines` of history. If `filename` is NULL, it tries truncating `~/.history`.

The question now is (1) What is the maximum number of lines to allow in the history file and (2) How many lines do we leave in the history after truncating? 

I wasn't sure, so I checked this with `zsh`, which I use in my terminal. The command history file on my machine is `.zshistory`. On my machine, the history file caps out at **3000 lines** and truncates the file by saving the most recent **2000 lines**.  

Let's make changes to address these issues.

```C
/* third_ex.c */
#include <stdlib.h>
#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <pwd.h>

#define MAX_LINES_HISTORY 3000
#define NLINES_HISTORY_FLUSH 2000

char *get_history_filename(void);
void create_history_file(const char *history_filename);


int main(){
    char *input;
    char* history_filename = get_history_filename();
    create_history_file(history_filename);
    /* Access global variable history_length to get size of history file at runtime */
    int nlines_history_file = history_length; 

    /* Add history file content to our current history list */
    read_history(history_filename);

    while (1) {
        /* Display prompt and read input */
        input = readline("prompt> ");

        /* Check for EOF */
        if (!input)
            break;

        /* Add input to history list */
        add_history(input);

        /* Add input to the history, but only if it is not blank content */
        if (*input){
            append_history(1, history_filename);
            nlines_history_file++;
        }

        /* Check if we need to flush the history file */
        if (nlines_history_file > MAX_LINES_HISTORY){
            if (history_truncate_file(history_filename, NLINES_HISTORY_FLUSH) != 0)
                printf("Couldn't truncate the history file");
            nlines_history_file = NLINES_HISTORY_FLUSH;
        }

        /* Print the input with an exclamation point. */
        printf("%s!\n", input);

        /* Free buffer that was allocated by readline */
        free(input);
    }
    return 0;
}

```

This program will now behave desirably, addressing all of our previous concerns. 

# A final nitpick: Consecutively duplicate commands
One bad thing about our program is that if the user enters the same exact command twice in a row, we store it twice in a row. Ideally, we should only store it once in the history file. This is how most history files behave. 

To do this, we need to track a variable `char *last_command`. We can initialize this variable by setting it equal to the last command recorded in the history file, making it `NULL` if the file is empty. 

To get the last command in the history file, we use the following function:

* `HIST_ENTRY *history_get(int offset)` will return a pointer to struct `HIST_ENTRY` whose position is at `offset`.  

Thus, after calling `read_history(history_filename)`, the history entry `history_get(history_length)` will correspond to the last command in the history file. We can obtain the contents of the last command by asking for `history_get(history_length)->line`.

The code to initialize `last_command` is:

```C
...
    /* Add history file content to our current history list */
    read_history(history_filename);

    /* Last recorded command in the history file */
    char *last_command = NULL;
    if (history_get(history_length) && history_get(history_length)->line)
        last_command = history_get(history_length)->line;
...
```
Notice that `last_command` is `NULL` if the history file is empty.

To avoid recording consecutively duplicate entries in our history file, we need to compare `last_command` with `input` using `strcmp`, and we need to change the part of the program that updates the history file. We cannot pass a null pointer to `strcmp`, so we must first check if it is `NULL` as below.

```C
...
    /* Add input to history list */
    add_history(input);

    /* Add input to the history file, but only if
     * (1) last_command is NULL or
     * (2) strcmp(last_command, input) != 0 (don't record duplicates) */
    if (!last_command || strcmp(last_command, input)) {
        if (append_history(1, history_filename) != 0)
            printf("Error: Couldn't append history to the history file.");
        nlines_history_file++;
        /* Update contents of last_command with current input */
        if (last_command)
            free(last_command);
        last_command = malloc(strlen(input) + 1);
        if (last_command == NULL)
            printf("Couldn't allocate memory to update the last command.");
        strcpy(last_command, input); /* Necessary because input is freed at end of loop */
    }
...
```

# The final program 
The final program that solves all of our chicken-and-egg problems and edge cases is as follows. Notice that we refactored our last change into a separate function `update_history_file`. 

```C
#include <stdlib.h>
#include <stdio.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <pwd.h>

#define MAX_LINES_HISTORY 3000
#define NLINES_HISTORY_FLUSH 2000

char *get_history_filename(void);
void create_history_file(const char *history_filename);
void update_history_file(const char *history_filename, char *last_command, char *input);
int nlines_history_file = 0;

int main(){
    char *input;
    char* history_filename = get_history_filename();
    create_history_file(history_filename);
    /* Access global variable history_length to get size of history file at runtime */
    nlines_history_file = history_length; 

    /* Add history file content to our current history list */
    read_history(history_filename);

    /* Last recorded command in the history file */
    char *last_command = NULL;
    if (history_get(history_length) && history_get(history_length)->line)
        last_command = history_get(history_length)->line;

    while (1) {
        /* Display prompt and read input */
        input = readline("prompt> ");

        /* Check for EOF */
        if (!input)
            break;

        /* Add input to history list, update our history file */
        add_history(input);
        update_history_file(history_filename, last_command, input)

        /* Print the input with an exclamation point. */
        printf("%s!\n", input);

        /* Free buffer that was allocated by readline */
        free(input);
    }
    return 0;
}

/* Get the full path of the history filename */
char *get_history_filename(void){
    /* First get the home directory of the user */
    char *homedir;
    if ((homedir = getenv("HOME")) == NULL)
        homedir = getpwuid(getuid())->pw_dir;

    /* Create the full path */
    size_t needed = snprintf(NULL, 0, "%s/.my_history", homedir);
    char *filename = malloc(needed + 1);
    if (filename == NULL)
        printf("Cannot allocate char pointer for history filename");
    return filename;
}

/* Create the history file if it doesn't exist */
void create_history_file(const char *history_filename){
    if (access(filename, F_OK) != 0) {
        FILE *hist_file = fopen(history_filename, "w");
        if (hist_file == NULL) {
            fprintf(stderr, "Can't open file %s", history_filename);
            exit(1);
        }
        fclose(hist_file);
    }
}

/* Update the history file, making sure not to record duplicates */
void update_history_file(const char *history_filename, char *last_command, char *input){
    /* Add input to the history file, but only if
    * (1) last_command is NULL or
    * (2) current executed command is different from the last command (don't record duplicates) */
    if (!last_command || strcmp(last_command, input)) {
        if (append_history(1, history_filename) != 0)
            printf("Error: Couldn't append history to the history file.");
        nlines_history_file++;
        /* Update contents of last_command with current input */
        if (last_command)
            free(last_command);
        last_command = malloc(strlen(input) + 1);
        if (last_command == NULL)
            printf("Couldn't allocate memory to update the last command.");
        strcpy(last_command, input); /* Necessary because input is freed at end of loop */
    }

    /* Check if we need to flush the history file */
    if (nlines_history_file > MAX_LINES_HISTORY){
        if (history_truncate_file(history_filename, NLINES_HISTORY_FLUSH) != 0)
            printf("Couldn't truncate the history file");
        nlines_history_file = NLINES_HISTORY_FLUSH;
    }
}

```

As we can see, GNU Readline provides the tools to implement a typical REPL shell that behaves just like the ones we use every day. Hopefully this code can save someone the time from having to rethink chicken-and-egg problems and other edge cases.

