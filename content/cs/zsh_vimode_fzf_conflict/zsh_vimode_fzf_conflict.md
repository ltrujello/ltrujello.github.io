<!-- title: fzf tab completion failure due to Zsh-Vi-Mode-->
<!-- syntax_highlighting: on -->
<!-- preview_img: /cs/zsh_vimode_fzf_conflict/imgs/correct_result.png -->
<!-- date: 2021-10-29 -->

In upgrading my terminal setup, an emotional and cathartic ritual I occassionally perform, I found that [fzf,](https://github.com/junegunn/fzf) a fuzzy finder for the command line, was failing to work. Specifically, fzf has a great feature where you can type something, append `**` to the end, and then hit tab, and it will begin a live search that you can continue typing characters in to speed up the search. You can mispell things and it still shows relevant stuff. And you can append this to pretty much any command, like `vim file**`. This makes it fast to execute commands whos arguments consist of files. Hence you can quickly find nested files and avoid typing, cd-ing and ls-ing. It looks like this normally.

<img src="/cs/zsh_vimode_fzf_conflict/imgs/correct_result.png" style="margin: 0 auto; display: block; width: 50%; height: auto;"/>

There's a cool [YouTube video](https://www.youtube.com/watch?v=qgG5Jhi_Els) showing many features of fzf. 

What was happening for me was instead the entire search result was being pasted into my terminal input, and no menu appeared. As a result I had a bunch of junk in my stdin.   

At first I had no idea if I was misunderstanding the usage, if I messed up the installation, if the most recently commit wasn't stable, etc. This was pretty annoying since no amount of keywords or combinations of relevant terms lead to a successful Google search of the issue. I had to tear apart my configure scripts and `.*rc` files, try fzf on a different machine, and run different tests to try to recreate and eliminate the mysterious issue.

After some time I pinpointed the issue to be due to a plugin I really liked, [zsh-vi-mode](https://github.com/jeffreytse/zsh-vi-mode), which I certainly didn't expect. I figured there was just something wrong with my own fzf or I messed up my `.zshrc` file. So I crawled through the Github issues of the respective repos to then find this [issue](https://github.com/jeffreytse/zsh-vi-mode/issues/24) that discusses it, which had the solution. 

Since this ate a lot of my time, I thought I would share this and the solution since I and some other poor soul on Github had to do a lot of debugging. Of course one can use another zsh vi implementation, but the only other competitor is the vi plugin from [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh/blob/master/plugins/vi-mode/vi-mode.plugin.zsh) which is pretty bad.

## The solution
The issue was basically that `zsh-vi-mode` was overwritting keyboard bindings at the very end of when the command line was loading. This is a bit of a bad design because it causes a confusing issue like this, and you really shouldn't be silently overwritting keyboard bindings that your users may or may not be using. The author of [zsh-vi-mode,](https://github.com/jeffreytse/zsh-vi-mode) who is pretty responsive to issues, addressed this by creating a setting `ZVM_INIT_MODE` that solves the problem, so that it basically prioritizes yours and other programs keybindings. 

If you use [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh/blob/master/plugins/vi-mode/vi-mode.plugin.zsh) for its themes and plugin management, then you need to add this setting right before you call the plugins. 
```bash
...
ZVM_INIT_MODE=sourcing
plugins(git
        zsh-vi-mode
        ...)
...
```
If you don't use [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh/blob/master/plugins/vi-mode/vi-mode.plugin.zsh), then you probably call `source some/path/to/zsh-vi-mode.sh` somewhere in your `.zshrc`. Hence enable the setting to `sourcing` right before that. 

