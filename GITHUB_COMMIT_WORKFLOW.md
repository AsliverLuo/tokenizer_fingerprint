# GitHub 仓库提交流程

本文档适用于将当前项目文件夹上传到 GitHub，并在后续持续提交更新。

当前项目路径：

```bash
/mnt/vos-79jtuvax/luoqihang/tokenizer-fingerprint-bcs-only_developing/tokenizer_fingerprint_simplified_20260730
```

---

## 1. 进入项目目录

```bash
cd /mnt/vos-79jtuvax/luoqihang/tokenizer-fingerprint-bcs-only_developing/tokenizer_fingerprint_simplified_20260730
```

---

## 2. 检查当前 Git 状态

```bash
git status
```

查看当前 Git 仓库根目录：

```bash
git rev-parse --show-toplevel
```

如果输出不是当前项目目录，而是上级目录，说明当前文件夹可能处在父级 Git 仓库中。若你希望把当前文件夹作为一个独立 GitHub 仓库，建议重新初始化当前目录的 Git 仓库。

---

## 3. 首次上传到 GitHub

### 3.1 初始化 Git 仓库

如果当前目录还不是独立 Git 仓库，可以执行：

```bash
git init
git branch -M main
```

如果当前目录已经被父级 Git 仓库管理，并且你确认要把当前目录改成独立仓库，可以先移除当前目录内的 `.git`，再重新初始化：

```bash
rm -rf .git
git init
git branch -M main
```

> 注意：执行 `rm -rf .git` 前请确认当前目录就是你要单独上传的项目目录。

---

### 3.2 创建 `.gitignore`

建议在项目根目录创建 `.gitignore`，避免提交缓存、环境文件、临时文件等：

```bash
cat > .gitignore <<'GITIGNORE'
__pycache__/
*.py[cod]
.venv/
.env
.env.*
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.DS_Store
*.log
.agents/
.codex/
GITIGNORE
```

---

### 3.3 检查是否有大文件或敏感信息

检查大文件：

```bash
find . -type f -size +50M
```

检查可能的敏感字段：

```bash
grep -R "api_key\|secret\|token\|password" . --exclude-dir=.git
```

如发现密钥、密码、token 等内容，先删除或改用环境变量后再提交。

---

### 3.4 添加文件并提交

```bash
git add .
git commit -m "Initial commit"
```

如果提示没有配置用户名和邮箱，执行：

```bash
git config --global user.name "你的GitHub用户名"
git config --global user.email "你的邮箱"
```

然后重新提交：

```bash
git commit -m "Initial commit"
```

---

### 3.5 在 GitHub 创建远程仓库

方式一：在 GitHub 网页创建空仓库。

示例仓库地址：

```text
https://github.com/你的用户名/tokenizer_fingerprint_simplified.git
```

创建时建议不要勾选：

- Add a README file
- Add .gitignore
- Choose a license

因为本地项目已经有文件，直接推送即可。

方式二：如果安装了 GitHub CLI，可以直接创建：

```bash
gh repo create tokenizer_fingerprint_simplified --private --source=. --remote=origin --push
```

如果要创建公开仓库：

```bash
gh repo create tokenizer_fingerprint_simplified --public --source=. --remote=origin --push
```

---

### 3.6 绑定远程仓库

如果使用 HTTPS：

```bash
git remote add origin https://github.com/你的用户名/tokenizer_fingerprint_simplified.git
```

如果使用 SSH：

```bash
git remote add origin git@github.com:你的用户名/tokenizer_fingerprint_simplified.git
```

如果已经存在 `origin`，需要改地址：

```bash
git remote set-url origin https://github.com/你的用户名/tokenizer_fingerprint_simplified.git
```

查看远程仓库地址：

```bash
git remote -v
```

---

### 3.7 推送到 GitHub

```bash
git push -u origin main
```

首次推送成功后，后续可以直接使用：

```bash
git push
```

---

## 4. 日常提交更新流程

每次修改代码后，按以下步骤提交：

### 4.1 查看改动

```bash
git status
```

查看具体差异：

```bash
git diff
```

---

### 4.2 添加改动文件

添加全部改动：

```bash
git add .
```

只添加指定文件：

```bash
git add 文件路径
```

---

### 4.3 提交改动

```bash
git commit -m "说明本次修改内容"
```

示例：

```bash
git commit -m "Update README and add evaluation scripts"
```

---

### 4.4 推送到 GitHub

```bash
git push
```

如果是第一次推送当前分支：

```bash
git push -u origin main
```

---

## 5. 查看当前 GitHub 账号

如果使用 GitHub CLI：

```bash
gh auth status
```

查看当前登录用户名：

```bash
gh api user --jq .login
```

查看 Git 提交身份：

```bash
git config --global user.name
git config --global user.email
```

查看当前仓库提交身份：

```bash
git config user.name
git config user.email
```

---

## 6. 重新绑定 GitHub 账号

退出旧账号：

```bash
gh auth logout -h github.com
```

登录新账号：

```bash
gh auth login -h github.com -p https -w
```

登录后确认账号：

```bash
gh api user --jq .login
```

修改 Git 提交身份：

```bash
git config --global user.name "新GitHub用户名"
git config --global user.email "新邮箱"
```

修改当前仓库远程地址：

```bash
git remote set-url origin https://github.com/新用户名/仓库名.git
```

然后重新推送：

```bash
git push -u origin main
```

---

## 7. 常见问题

### 7.1 `remote origin already exists`

说明已经绑定过远程仓库。可以改用：

```bash
git remote set-url origin https://github.com/你的用户名/仓库名.git
```

---

### 7.2 `Authentication failed`

可能是 GitHub 登录账号或 token 失效。

可以重新登录：

```bash
gh auth logout -h github.com
gh auth login -h github.com -p https -w
```

然后再推送：

```bash
git push
```

---

### 7.3 `src refspec main does not match any`

通常表示还没有提交，或者当前分支不是 `main`。

检查分支：

```bash
git branch
```

如果还没有提交：

```bash
git add .
git commit -m "Initial commit"
```

然后推送：

```bash
git push -u origin main
```

---

### 7.4 GitHub 仓库已有 README 导致 push 失败

如果 GitHub 远程仓库已经初始化了 README，可能需要先拉取：

```bash
git pull origin main --allow-unrelated-histories
```

解决冲突后再提交并推送：

```bash
git add .
git commit -m "Merge remote repository"
git push
```

---

## 8. 推荐完整命令模板

将当前项目作为新仓库上传时，可以参考：

```bash
cd /mnt/vos-79jtuvax/luoqihang/tokenizer-fingerprint-bcs-only_developing/tokenizer_fingerprint_simplified_20260730

git init
git branch -M main

git add .
git commit -m "Initial commit"

git remote add origin https://github.com/你的用户名/tokenizer_fingerprint_simplified.git
git push -u origin main
```

后续日常提交：

```bash
cd /mnt/vos-79jtuvax/luoqihang/tokenizer-fingerprint-bcs-only_developing/tokenizer_fingerprint_simplified_20260730

git status
git add .
git commit -m "说明本次修改内容"
git push
```
