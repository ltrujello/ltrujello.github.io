<!-- title: Arch Linux Install Guide and Dell XPS 9710 Experience -->
<!-- syntax_highlighting : on -->
<!-- date: 2021-12-20  -->
This is an Arch install guide where the machine in question is a new Dell XPS 17 9710, although the guide should apply to other machines as well.

If you are on a laptop, make sure the machine is charging before you start. You'll be looking at a terminal screen and so you won't be able to see your battery. (Yes I'm speaking from experience...)

## Change boot settings 
Restart your computer to change your UEFI settings and to disable secure boot. You should also give highest booting priority to your installation media if you're booting from a USB. 

On my machine, I did this by hitting F12 as soon as the Dell logo appeared. I also had to be extremely quick in hitting F12 since otherwise it would try booting Windows, give me a blue screen, and restart.

If you don't see your USB device, you have to restart and try again.

**Tip**: I got stuck in a loop where my usb device containing my live arch image would not appear in the menu for booting priority. One thing that worked was powering off the machine, powering it on with USB removed, and *then* plugging in the USB right before the Dell logo appeared. Doing this got me out of that loop.

Also, there may be two of the same devices listed for some reason. I just put both at the top.

## Booting in the live image
After you boot to the live image, you'll be met with a tty running a zsh prompt.
```
root@archiso~#
```

The first time I did this, I couldn't type any coherent input into the tty since the stdin was being randomly flooded with error messages like  `SQUASHFS error: Unable to read fragment cache entry`. 

Online results said that my image must have been corrupted. I knew that wasn't the case (since I verified the image), but I knew my USB in question was a piece of crap in the past, so I flashed a newer one and that worked.

Following the arch guide, I ran 
```
ls /sys/firmware/efi/efivars
``` 
to verify that I was in UEFI mode. 

## Connect to wifi in the live image
Next, you need to connect to the internet. You do this by using `iwctl`, which will open a command prompt that looks something like 
```
[iwd]#
``` 
Run `help` to see your options and so you understand what's going on.

Next, run the command
```
[iwd]# device list
```
to see your network device list. Look for something like `wlan0`, or something starting with a `w`, to find your wifi device. For me that was `wlan0`.

Now scan networks with your wifi device via the command 
```
[iwd]# station wlan0 scan
```
Get the results of the scan via 
```
[iwd]# station wlan0 get-networks
```
and you will see your available wifi networks.

Connect to a wifi network of your choosing via 
```
[iwd]# station wlan0 connect <network_name>
```
Note that tab complete is available, in case you have a long network name.

Verify the connection by running the command 
```
ping archlinux.org
```

Keep in mind that this is *not* the same as performing the network configuration for your actual Arch installation (which you will do painfully later). In fact, your own installation of Arch doesn't even exist on your computer at this point.
The connection you have now will remain until you reboot. So it's important to download certain packages while you have this internet connection, which we will discuss later.

Next, I used the command 
```
timedatectl set-ntp true
``` 
to synchronize the system clock with the network.

# Configure Partitioning Table and Scheme
You now need to do three things with your disks.

* Identify your computer's disk (singular, assuming you're on a laptop with one physical storage disk of 256GB, 500GB,etc.) and examine the **partitioning table** and **scheme** 

* Create **partitions** of your disk to assign your linux file systems and boot (so your machine boots to your installation automatically, forever, when this is all over)

* **Format** the partitons you created

* **Mount** the partitons you created (so you can access your Arch installation from the live image you're currently running).


If you've never done any of this before, this part will (and should) take around an hour or two because you need to understand exactly what **partitioning** (tables, schemes), **formating**, and **mounting** are. If you are otherwise not careful, you can mess something up with a high likelihood (although you'll probably be fine even if you do). If you don't understand the terms above in bold, read up on them. I certainly did not know these terms when I approached this step and it really helped to learn these terms so I knew what exactly I was doing.

1. To find your computer storage disk that you will install Arch on, identify all of your disks and the existing partition of the disks. You can do this by running any of the commands.
```bash
fdisk -l  # Verbose list of storage devices and their partitions
lsblk  # Lists the partitions in a visual tree format
parted -l  # Lists the partition tables, size, and filesystems 
```
To keep things simple, stick to `fdisk -l`. It will always tell you everything you need to know, and you will use `fdisk` later in the installation too (the less commands to remember, the better).

These commands will show you storage devices like 

* `/dev/sda` which corresponds to a SATA disk 

* `/dev/nvme0n1`which corresponds to an NVMe disk

* `/dev/mmcblk0` which corresponds to a eMMC disk

For me, on a new laptop, my disk was `/dev/nvme0n1` and I had 5 partitions like 
```
/dev/nvme0n1p1 EFI
/dev/nvme0n1p2 Microsoft Data
/dev/nvme0n1p3 Windows Recovery 
/dev/nvme0n1p4 ...
/dev/nvme0n1p5 ...
```
You'll probably also see your USB device from which you are booting you live image from, which probably is something like `/dev/sda`.

2. Decide on your partitioning scheme.
There are two choices for what partitioning scheme you want to use: Master Boot Record (MBR) and GUID Partition Table (GPT). To decide between the two, see [this discussion](https://wiki.archlinux.org/title/partitioning#Choosing_between_GPT_and_MBR). Long story short, you probably want GPT.
3. Before you actually partition your disk, back up your current partitioning table and scheme. To do this, use the name of your storage device. For me, that was `/dev/nvme0n1`. Therefore, I ran the command 
```
sfdisk -d /dev/nvme0n1 > nvme0n1.dump
```
where of course you should supply your correct disk. Now if you need to restore your partitioning table and scheme you can run
```
sfdisk /dev/nvme0n1 < nvme0n1.dump
```

4. Now partition the actual disk. There are a number of commands you can use, but `fdisk` does the job. Since in my case my computers storage device was `/dev/nvme0n1`, I ran
```
fdisk /dev/nvme0n1
```
Of course, run the command with your storage device that you found. 
Running this command will load a command prompt that looks like this 
```
Command (m for help):
```
and you will use this command prompt to create the partitions. This command makes no changes to the drive's partitioning scheme until you tell it to explictly with the `w` command.  

You should press `m` to see the options and usage. 

5. See if you have an existing EFI partition. The Arch docs recommend not formatting an existing EFI partition, since you'll just have to make one again anyways, and it could have something useful for you later on. In my case 
```
/dev/nvme0n1p1
```
was an already existing EFI partition, and as my partitioning scheme was GPT, I simply left it alone. I then deleted the other partitons since none of them were of importance. 

6. You will now perform the actual partitioning depending on which scheme you chose. In my case my disk was `nvme0n1`, I chose GPT, and I decided on the following partition layout:
```
/dev/nvme0n1p1 200MI EFI System

/dev/nvme0n1p2 372GB Home filesystem 

/dev/nvme0n1p3 60GB  Root filesystem

/dev/nvme0n1p4 48GB  Swap
```
That is, I wanted to separate my root and home file systems into different disk partitions. Many people online recomend this, but most GUI-installers for Linux distributions will by default only make you one root partition in which your home directory resides.

I also decided on a swap partition, which is used for the RAM and suspending the machine. I decided on 48GB for the swap size via the formula `swap_size = 1.5 * <your_RAM>` which in my case was 32GB.

Since a common paritioning is to have partitions for `/`, for operating system files and booting, `/home`, for my own user files, and a third partition for RAM swap. I did research into seeing if that was worth it. Usually everything is just lumped into `/`. I ended up deciding on the following partitioning table decided on the partition table:

For more on how exactly you can partition your disk, here are some references.
To decide on your own partitioning scheme, here's a [great reference](https://www.dell.com/support/kbdoc/en-us/000131456/the-types-and-definitions-of-ubuntu-linux-partitions-and-directories-explained) by Dell about common partitioning schemes and sizes in Linux. [Here](https://bbs.archlinux.org/viewtopic.php?id=260759) is also a helpful and funny Arch linux discussion on the topic of partition sizes.



To achieve this desired partitioning, I needed to perform disk partitioning in a certain order, and you will too. This is because you probably want your home directory (or root directory, if you're doing just one root partition) to take up the largest amount of space possible around your swap file and EFI system. 
For my specific partitioning, I performed the following steps.

First, I deleted the pre-existing partitions that I didn't want using the `d` command. These were some useless partitions created by Microsoft Windows (useless to me because I hate Windows, probably useful to you if you want to dual boot).

## Partition for `/`
Since I chose to separate my root and home into different partitions, I first created the partition where my root file system would reside, since it only needs about 30GB. I did this with the `n` command. 

* For `Partition Number`, I entered 2, becaue `/dev/nvme0n1`, my EFI system, was already taken.

* For `First Sector`, I simply hit enter for the default value.

* For `Second Sector`, I entered `+60GB` to give my root file system 60GB. 

* This created the partition, but I wanted to the partition type to be `Linux root (x86-64)` and not `Linux filesystem`. To do this, I used the `t` command, and examined the list of partition types.

## Parition for the swap
Next I created a swap partition again with `n` command.

* For `Partition number` I did `4`.

* For `First sector` I again hit enter.

* For `Second sector` I entered `+48GB`.

* Then I used the `t` command to change this partition type to `swap`.

## Partition for `/home`
Lastly, I created the partition for my home file system for last as I wanted it to take up as much space as possible around my other partitions. 

* For `Partition number` I did `3`.

* For `First sector` I hit enter.

* For `Second sector` I also hit enter. The default is to take up the rest of the space.

* I was then happy with the partition being of type `Linux filesystem`. It really doesn't matter in this case. 

Finally, I hit the `w` command to write my partitions, and verified this is what I wanted with the `fdisk -l` command.

A list of resources:

- [arch linux `fdisk` docs](https://wiki.archlinux.org/title/fdisk) for understanding what the command can do

- [here's](https://www.howtogeek.com/106873/how-to-use-fdisk-to-manage-partitions-on-linux/) a basic guide on using `fdisk`. It is outdated, but it's helpful since it has screenshots.

- [A list of common partition types](https://en.wikipedia.org/wiki/GUID_Partition_Table#Partition_type_GUIDs) from archlinux.org

- [A good question](https://unix.stackexchange.com/questions/557470/what-are-the-differences-between-linux-filesystem-linux-server-data-linux-root) about difference between partition types

# Format the Partitions
Now you need to format your partitions in order to prepare them for installing linux filesystems onto them. In my case, I did this with the following commands. 
```bash
mkfs.ext4 /dev/nvme0n1p2 # Format the root 
mkfs.ext4 /dev/nvme0n1p3 # Format the home directory 
mkswap /dev/nvme0n1p4 # Initialize the swap partition
```
And I did nothing with my existing EFI partition. If I tried to format it, it would just erase everything on it.

# Mount the partitons
Next, mount your partitions onto the directory `/mnt`. This will allow you to actually perform the Arch linux installation on your disk by giving yourself a point of reference to your disk.

The Arch linux docs do not state this, but it is actually very important and critical that you mount your partitions into a very specific order.
In any case, you first need to do `root`. You can't, for example, mount the `home`, then mount the `root`; you will lose access to data in `home`. You instead need to mount the `root`, then the `home`. 

Since the EFI boot system and `home` do not collide, the order of mounting these two doesn't matter. 

So in my particular case, I performed the following commands in this order to achieve a correct mounting. 
```bash
mount /dev/nvme0n1p2 /mnt # Mount the root
mkdir /mnt/boot # Create the boot directory
mount /dev/nvme0n1p1 /mnt/boot # Mount the EFI to boot
mkdir /mnt/home  # Create the home directory
mount /dev/nvme0n1p3 /mnt/home # Mount the home 
swapon /dev/nvme0n1p4 # Designate my swap partition 
```
You can see these commands are successful by runnning `ls /mnt`. To
check that the swap is successful, run `swapon --show` to show the swap devices. 

To see all of the mount points after you're done visually, run `findmnt`. 

Also note that if you decided to also create a separate partition for your `/home` directory, you may find a `lost+found` file sitting there. That's normal, according to [this](https://unix.stackexchange.com/a/125863) question and answer. 

# Remainder set up
Next install Arch linux onto your disk via the command 
```
pacstrap /mnt base linux linux-firmware
```
Create an fstab file via the command
```
genfstab -U /mnt >> /mnt/etc/fstab
```
Enter your newly installed Arch linux via the command
```
arch-chroot /mnt
```
At this point, you are in your Arch linux install and **you have internet**. You will lose this connection if you reboot. So it is very wise to download anything you think you may possibly need (`vim`, `curl`, `git`, etc). On top of this you'll download some packages later in this guide.  

In the new system, update the timezone information via
```
ln -sf /usr/share/zoneinfo/<region>/<city> /etc/localtime
```
You can use tab-complete to easily find your region and city. Then run the
```
hwclock --systohc
```
Next, you need to edit files, so unless you want to torture yourself using `nano`, download `vim` or whatever.
```
pacman -Syu vim
```
Next, edit `/etc/locale.gen` and uncomment `en_US.UTF-8 UTF-8`. Create the file `/etc/locale.conf` and write `LANG=en_US.UTF-8`. Finally, run the command 

```
locale-gen
```

Next, create your system hostname by creating and writing in the file `/etc/hostname` file.

# Download essential things
The next step will require you to configure a boot loader, so that your computer acts like a normal computer and boots to Arch linux automatically when it powers on. This requires installing `grub`. However, we also need some other packages for later, so we are just going to install a lot of things all at once. 

Run the command 
```
pacman -Syu grub efibootmgr mandb networkmanager network-manager-applet wireless_tools wpa_supplicant mtools dosfstools base-devel linux-headers
```
Add anything else you want. Of particular importance:

* `grub` and `efibootmgr` are packages that will help you set up your boot loader.

* `mandb` allows you to use the `man` command, which is essential so you can investigate commands that you will use later on.

* `network-manager` is what you will use to start the `NetworkManager.service` in order to connect to wifi. 

All other tools are extra things that may come in handy. 

# If you forget to download something
Soon, you will restart your computer and boot into your installation, at which point you will have no internet. If you forget to download something once you're restarted, you can try connecting with ethernet. Your other option will be to

* Power off your computer, go into recovery mode to reassign your live USB the highest boot priority

* Reboot back into your live image

* Reconnect to your wifi using `iwctl`
 
* Remount your partitions onto `/mnt` (again, in the right order)

* `arch-chroot` back into your arch installation

and then proceed to use your wifi connection to download whatever you need.

# Boot loader
A common boot loader is GRUB, which I settled on for my installation. One thing I needed to configure for my particular machine, and you probably will too, is to configure microcode updates for your system's firmware. 

Since I had an intel chip, I followed the Arch linux docs by running the command 
```
pacman -Syu intel-ucode
```
Once you have your firmware packages installed, you need to install grub onto your Arch linux insatllation. In my case, I needed to put grub in `/boot`, which is the typical location for the boot loader. This is also the location of where I mounted my pre-existing EFI system. Thus the appropriate command for this is
```
grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB
```
Next you need to set up a `grub.cfg` file. Again, it's important to have installed the fimware `intel-ucode` first before configuring the grub file. The command to do that is 
```
grub-mkconfig -o /boot/grub/grub.cfg
```

# Create a root password and user
Next, you need to create a password for the root account. Do that by running 
```
passwd
```
If you forget to do this and restart your computer, then you will have to boot back into your Arch live image, remount your disk partitions, `arch-chroot` back into your system, and then create a password.

# Reboot the system 
You can now restart your computer and hopefully it should automatically boot to your installed Arch system. You can of course do this with 

```
reboot
```

# Configure the network
Congrats! You now have Arch Linux. Now hell begins (not really). 

You next need to configure your network which can be anything from seamless to a huge pain in the ass. There are various network packages you can choose. In my case, I eventually settled on Network Manager.

First identify what wifi card your computer has. To do this, run the command
```
lspci -v | grep Network
```
For me, it returned something like 
```
Network controller: Intel Corporation Tiger Lake PCH CNVi WiFi
```
which I then googled to determine what specific actions I needed to do for this specific wifi card. You should do the same with whatever card shows up for you.

Your goal is to first have your system recognize that you even have a wifi card. That is, you want a wireless device to appear under 
```
ip link show
```
and you want to have an IP address under
```
ip address show 
```
For me, my wifi card did not appear when I ran `ip link show`. To make it appear, I needed to investigate why my firmware was failing to load via the command
```
dmesg | grep firmware
```
and as well as looking at the system journal via `journalctl`. Eventually, I found out that I needed to run this set of commands to load my firmware
```
sudo modprobe -r iwlnvm
sudo modprobe -r iwlwifi
sudo modprobe iwlwifi
```
Once I manually loaded my firmware, my wifi device appeared under the `ip link show` command as device `wlan0`. I then ran 
```
ip link set wlan0 up
```
Once I set up the device, I needed to **enable** and **start** the service `NetworkManager` which I did with `systemctl`. 

Finally, I connected to my wifi network with `NetworkManager`. (I tried using `iwctl`, but that didn't happen to find my wifi device even though it showed up in `ip link show`. I also tried `wpa_supplicant`, but that also did not work.)
I ran the commands [here](https://wiki.archlinux.org/title/NetworkManager).

After this, my internet connection worked. In rebooting my internet connection continued to work out of the box. 

## Frequently disconnecting wifi
One error I did have was that my wifi randomly disconnected and my computer's local IP address was being reassigned on a matter of minutes. I only noticed this because I noticed my SSH connections were being randomly interrupted. Eventually I found [here](https://bbs.archlinux.org/viewtopic.php?id=230992) that it was happening because I had both `dhcpcd.service` and `NetworkManager.service`running. I disabled and stopped `dhcpcd.service` and my wifi continued to behave.

# Where to go from here

## Sound 
To get the sound working, I had to install the package `sof-firmware`, since the Dell XPS 9710 is new, and then reboot my computer. You should do some research to see what sound firmware your computer needs.

To actually change the sound settings, I installed the package `alsa-utils`. This package installed the command `amixer`, the manpages of which are extremely clear on how to use it to adjust your volume. You can read more about `alsa` [here](https://wiki.archlinux.org/title/Advanced_Linux_Sound_Architecture#Installation).

## Brightness
To control the screen brightness, I used `brightnessctl`. The manpages are very clear on how to use this. 

## Mapping volume and brightness control keys
Since I settled on `bspwm` as a window manager, I used `sxhkd` to manage my keybindings. In order to actually map my keys, I needed to figure out what their names were, which I did via [this post](https://www.reddit.com/r/bspwm/comments/foop05/for_sxhkd_with_bspwm_how_to_view_list_of_keyboard/flga1mw/) which suggested using `xev`.

After finding out the names of my keys I then appended this to my `sxhkdrc` file:
```
# brightness control
XF86MonBrightnessDown
        brightnessctl set 5%-

XF86MonBrightnessUp
        brightnessctl set 5%+
        
# volume control
XF86AudioMute
    amixer sset Master unmute

XF86AudioLowerVolume
    amixer set Master 5%-

XF86AudioMute
    amixer set Master 5%+
```


## Package management

At this point, you are probably going to customize your set up further via downloading packages from the internet. For example, you'll probably get yourself a desktop environment or window manager like bspwm. Reading up on the following links should help you with that. 

* [Arch User Repository](https://wiki.archlinux.org/title/Arch_User_Repository)
* [Arch package mirrors](https://wiki.archlinux.org/title/mirrors)
 





