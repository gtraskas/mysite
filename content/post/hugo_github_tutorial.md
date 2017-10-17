---
title: "How to create your blog/site with Hugo and GitHub"
date: 2017-10-17
tags: ["github", "hugo", "blog"]
draft: false
---

A step-by-step guide for macOS 

### Create two new repositories in [GitHub](https://github.com)
First sign in your profile page or make a new profile, if you haven't already one. Then:

1. Go to `Repositories` page and click to the `New` icon to create a new repository with the name of your project (e.g. `MySite`) and initialize with `README`. This repository will contain Hugo’s content and other source files.

2. Similarly, create a second repository with name `USERNAME.github.io`, which will contain the fully rendered version of your Hugo website. Open `terminal` in your working directory and clone this repository into your local directory.
```
$ git clone https://github.com/USERNAME/USERNAME.github.io.git
```
A new directory with the same name was created.

### Install Hugo
Install the latest Hugo version in your machine following this [guide](https://gohugo.io/getting-started/installing).

### Create your site
From `terminal` make sure you are still in your working directory and create a new directory for your website content and source files. Name this folder same as your project repository, i.e. `MySite`. Then go inside your new directory and create your new site with Hugo.
```
$ mkdir MySite
$ cd MySite/
$ hugo new site .
```
Initialize an empty Git repository, add your project repository, and fetch from your GitHub Project repository.
```
$ git init
$ git remote add origin https://github.com/USERNAME/MySite.git
$ git pull origin master
``` 
Add `public/` to `.gitignore` instead of completely remove this folder. A gitignore file specifies intentionally untracked files that Git should ignore.
```
$ nano .gitignore
```
Write `public/` to the opened file, save and exit. Then add file contents to the index, record changes to the repository, and update remote refs along with associated objects.
```
$ git add .
$ git commit -m "Initial commit"
$ git push -u origin master
```

### Install a Hugo theme
First, make sure you work in your project folder `MySite`. Got to the [Hugo Themes](https://themes.gohugo.io) website, select a theme and install it following the guide. For example, for the [Minimal](https://themes.gohugo.io/minimal/) theme you do the following:
```
$ git submodule add https://github.com/calintat/minimal.git themes/minimal
$ git submodule init
$ git submodule update
$ git submodule update --remote themes/minimal
```
After installation, take a look at the `exampleSite` folder inside `themes/minimal`. To get started, copy the `config.toml` file inside `exampleSite` to the root of your Hugo site (your project directory, i.e. `MySite`)
```
$ cp themes/minimal/exampleSite/config.toml .
```
You can now edit this file and add your own information.

### Test your website locally
Make your website work locally
```
$ hugo server
```
 and open your browser to http://localhost:1313. Press Ctrl+C to kill the server.

### Create content
Create content for your website with your favourite editor, e.g. Visual Studio Code, editing the `config.toml` and the files in the `content` directory inside your project folder `MySite`. Test your site locally everytime you want to check your changes. Don't forget to kill the server when you finish.

### Deploy your site
Once you are happy with the results, you are ready to deploy your site. Make sure that you are in your project directory, `MySite` and run the following:
```
$ hugo -d ../USERNAME.github.io/
```
This command creates all the required files for your website. Now, it's time to go to your second directory which contains the fully rendered version of your Hugo website, add the newly created files to the index, commit the changes and send everything to your remote repository.
```
$ cd ../USERNAME.github.io/
$ git status
$ git add --all
$ git commit -m "Initial commit"
$ git push origin master
```
Done! In a few seconds you will be able to see your website.