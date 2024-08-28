# 🦜️🔗 LangChain AI21

This repository contains 1 package with AI21 integrations with LangChain:

- [langchain-ai21](https://pypi.org/project/langchain-ai21/)

## Initial Repo Checklist (Remove this section after completing)

This setup assumes that the partner package is already split. For those instructions,
see [these docs](https://python.langchain.com/docs/contributing/integrations#partner-packages).

In github

- [ ] Add integration testing secrets in Github (ask Erick for help)
- [ ] Add partner collaborators in Github (ask Erick for help)
- [ ] "Allow auto-merge" in General Settings 
- [ ] Only "Allow squash merging" in General Settings
- [ ] Set up ruleset matching CI build (ask Erick for help)
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes

Pypi

- [ ] Add new repo to test-pypi and pypi trusted publishing (ask Erick for help)

Slack

- [ ] Set up release alerting in Slack (ask Erick for help)

release:
/github subscribe langchain-ai/langchain-ai21 releases workflows:{name:"release"}
/github unsubscribe langchain-ai/langchain-ai21 issues pulls commits deployments
