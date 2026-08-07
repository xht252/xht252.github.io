---
# Leave the homepage title empty to use the site title
title: ''
summary: >
  Haotian Xue's personal homepage featuring research notes,
  open-source projects, and technical blogs on artificial intelligence,
  multi-agent reinforcement learning, robotics, world models,
  and autonomous systems.
date: 2026-08-01
type: landing

sections:
  # Hero
  - block: dev-hero
    id: hero
    content:
      username: me
      greeting: "Hi, I'm"
      show_status: true
      show_scroll_indicator: true
      typewriter:
        enable: true
        prefix: "I explore"
        strings:
          - "Artificial Intelligence"
          - "Multi-Agent Reinforcement Learning"
          - "Robotics and Autonomous Systems"
          - "World Models"
          - "Embodied AI"
          - "Open-Source Software"
        type_speed: 70
        delete_speed: 40
        pause_time: 2500
      cta_buttons:
        - text: View My Projects
          url: "#projects"
          icon: arrow-down
        - text: Read My Blog
          url: "/blog/"
          icon: document-text
    design:
      style: centered
      avatar_shape: circle
      animations: true
      background:
        color:
          light: "#fafafa"
          dark: "#0a0a0f"
      spacing:
        padding: ["6rem", "0", "4rem", "0"]

  # Featured Projects
  - block: portfolio
    id: projects
    content:
      title: "Featured Projects"
      subtitle: "Selected research, engineering, and open-source work"
      count: 0
      filters:
        folders:
          - projects
      buttons:
        - name: All
          tag: '*'
        - name: Artificial Intelligence
          tag: AI
        - name: Systems
          tag: Systems
        - name: Web
          tag: Web
        - name: Java
          tag: Java
      default_button_index: 0
      archive:
        enable: true
        text: "Browse All Projects"
        link: "/projects/"
    design:
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Tech Stack
  - block: tech-stack
    id: skills
    content:
      title: "Tech Stack"
      subtitle: "Tools I use for research, development, and experimentation"
      categories:
        - name: Languages
          items:
            - name: Python
              icon: devicon/python
            - name: C/C++
              icon: devicon/cplusplus
            - name: Java
              icon: devicon/java
            - name: Markdown
              icon: devicon/markdown
        - name: AI & Robotics
          items:
            - name: PyTorch
              icon: devicon/pytorch
            - name: ROS
              icon: devicon/ros
            - name: Jupyter
              icon: devicon/jupyter
        - name: Development
          items:
            - name: Git
              icon: devicon/git
            - name: GitHub
              icon: brands/github
            - name: Linux
              icon: devicon/linux
        - name: Web
          items:
            - name: Flask
              icon: devicon/flask
            - name: Vue.js
              icon: devicon/vuejs
            - name: JavaScript
              icon: devicon/javascript
    design:
      style: grid
      show_levels: false
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Education and Research Focus
  - block: resume-experience
    id: experience
    content:
      title: "Education & Research"
      date_format: "Jan 2006"
      items:
        - title: "Graduate Student in Software Engineering"
          company: "Sichuan University"
          company_url: "https://www.scu.edu.cn/"
          company_logo: ''
          location: "Chengdu, China"
          date_start: "2025-09-01"
          date_end: ''
          description: |2-
            * Research interests include multi-agent reinforcement learning, autonomous systems, world models, embodied AI, and robotics
            * Develop learning-based methods for intelligent decision-making and coordination
            * Maintain research code, experiments, technical notes, and open-source projects
        - title: "B.Eng. in Computer Science and Technology"
          company: "Civil Aviation University of China"
          company_url: "https://www.cauc.edu.cn/"
          company_logo: ''
          location: "Tianjin, China"
          date_start: "2021-09-01"
          date_end: "2025-06-30"
          description: |2-
            * Studied computer science, software engineering, computer networks, and intelligent systems
            * Completed projects involving C/C++, Java, web development, network programming, and system simulation
    design:
      columns: '1'
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Selected Publications
  - block: collection
    id: publications
    content:
      title: "Selected Publications"
      subtitle: "Research papers, preprints, and technical reports"
      text: ""
      filters:
        folders:
          - publications
        exclude_featured: false
      count: 4
      order: desc
      archive:
        enable: true
        text: "View All Publications"
        link: "/publications/"
    design:
      view: card
      columns: 2
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Recent Blog Posts
  - block: collection
    id: blog
    content:
      title: "Latest Posts"
      subtitle: "Research notes, technical tutorials, project records, and personal reflections"
      text: ""
      filters:
        folders:
          - blog
        exclude_featured: false
      count: 3
      order: desc
      archive:
        enable: true
        text: "Read All Posts"
        link: "/blog/"
    design:
      view: card
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Contact
  - block: contact-info
    id: contact
    content:
      title: "Get In Touch"
      subtitle: "Research collaboration, open-source development, and technical discussions"
      text: |-
        I am interested in discussing artificial intelligence, multi-agent systems,
        autonomous systems, robotics, world models, and open-source projects.

        Feel free to contact me by email or through GitHub.
      email: xuehaotian@stu.scu.edu.cn
      autolink: true
    design:
      columns: '1'
      background:
        color:
          light: "#f5f5f5"
          dark: "#08080c"
      spacing:
        padding: ["4rem", "0", "4rem", "0"]

  # Final Call to Action
  - block: cta-card
    content:
      title: "Explore My Notes and Projects"
      text: |-
        I use this website to organize research ideas, technical tutorials,
        development records, and reflections from my learning journey.
      button:
        text: "Visit the Blog"
        url: "/blog/"
        new_tab: false
    design:
      card:
        css_class: 'bg-gradient-to-br from-primary-200 via-primary-100 to-secondary-200 dark:from-primary-600 dark:via-primary-700 dark:to-secondary-700'
        text_color: dark
      background:
        color:
          light: "#ffffff"
          dark: "#0d0d12"
      spacing:
        padding: ["4rem", "0", "6rem", "0"]
---
