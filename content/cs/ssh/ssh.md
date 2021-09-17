<!-- title: SSH -->
<!-- syntax_highlighting: on -->

Some things I've learned about SSH that I'm writing down if I ever forget.

## SSH Logging and Configuration

* SSH activity is generally logged in a specific file. The specific location varies on each system, but it is usually easy to find. Some possible locations are `/var/log/auth.log` or ```/var/log/secure```.

* Your machine SSH settings for accepting connections and for performing the SSH protocal are usually in ```/etc/ssh/```. There will be a ```sshd_config``` file which specify your desired settings *as an SSH host*, and there will be a ```ssh_config``` file which specify your desired settings *as an SSH client*. These settings can also be configured per individual user.

* Your individual user settings and public/private keys are in ```$HOME/.ssh```. In this, you will find three files.

* * ```authorized_keys```: This contains a list of public keys belonging to users who are allowed to ssh into your user. 

* * ```known_hosts```: This is a simple list of the hosts you've connected to. For each host, you will see the user you connected to, the port you used, and the host's fingerprint.
A host is added after the first time you make a brand new connection with that the host. You do so by verifying the fingerprint hash that it presents to you when you first make a brand new connection. It's almost always a SHA256, unless the host for some weird reason specifically configured to present an MD5 hash. 

* * Finally, there is a `config` file that allows you to configure your connections to SSH hosts thatn you know (such as what keys to try).


## SSH Key Generation and Management
**Generating keys**
To generate a set of ssh_keys, you run the command
```sh
ssh-keygen -t [key_algorithm] -b [bit_size]
```
Here ```-t``` specifies the algorithm you can use and -b specifies the key bit size. After you run this you'll be asked to give a file name; if none, it enters a default. It will then ask you for a pass phrase.

**Add SSH Keys to the agent**
To add your SSH keys to your SSH-agent, run
```bash
ssh-add
```
This adds your private and public keys to the ssh-agent, which manages your computer's ssh keys across users. 

## Basic SSH Commands
**Simple SSH protocol**
```bash
ssh -i [private_key] -p [port] [user]@[ip_address]
```
You don't need the ```-i``` flag if you know the host and have configured the host with an `IdentityFile` parameter. Otherwise, ssh might try every single key in your .ssh directory, but this then leaks information to the server about what kind of keys you've got (you may not care). With that said, it's good practice to supply the correct key or configure the host.

**Transfer a file from current machine to the host**
```bash
scp -i [private_key] -P [port] [file_paths] [user]@[host_ip_address]:[destination_on_host]
```
Here, ```scp``` stands for secure copy protocol, but it just uses the SSH protocol. Note something one can easily forget is that ```scp``` takes in the port with a capital ```-P``` flag, while ```ssh``` uses a lower case ```-p```. 

**Transfer a file from the server to the current machine**
```bash
ssh-copy-id [public_key] [user]@[host_ip_address]
```
This command is nothing more than SSH-ing into the host via password and copying the contents of the public key to the host's `authorized_keys` file. Hence, do not accidentally supply your private key.

The only reason you would run this is if you've never given the host your public key before. Since the host doesn't have your public key, your only option of authentication is password authentication (if it already has your keys, it will simply do nothing and inform you of that). Therefore, if the host doesn't allow passwords, you'll need to find another way to get your public key on there.  

