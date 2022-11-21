# < pre-commit 적용 순서 >
# 1. git init(생략)
# 2. pip install pre-commit     ## pre-commit 설치
# 3. pre-commit install         ## .git/hooks/pre-commit 내용을 변경하는 명령어 : 최초 한번 실행
# 4. pre-commit run             ## .pre-commit-config.yaml 파일이 있다면 실행됨


# git add .
# git commit -m 'pre-commit_test'
## config 파일을 통해 black, isort, autoflake8 실행되어 적용됨
## 다시 git add와 commit을 실행하면 black,isort가 적용된 파일로 commit됨


# [ 수정 시 ]
# flake8 경우, setup.cfg파일에서 수정하기
## ignore에 무시할 오류 작성해두기
