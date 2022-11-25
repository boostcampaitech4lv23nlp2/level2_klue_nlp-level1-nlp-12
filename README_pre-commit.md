<<<<<<< HEAD
```
# [ Setting ]
installed isort, black, flake8, pre-commit
```

```
# [ pre-commit 실행 방법 ]
=======
# Setting
```
installed isort, black, flake8, pre-commit
```

# pre-commit 실행 방법
```
>>>>>>> master
- 테스트용(stage에 올라온 파일만 검사) :
git add [file] > pre-commit run

- 실행 방법(commit 실행시 교정, 검사 완료) :
git add . > git commit -m '[write the message]'
```
<<<<<<< HEAD
```
# [ custom file ] - 수정 파일
=======

# custom file - 수정 파일
```
>>>>>>> master
- pyproject.toml : isort, black
- setup.cfg : flacke8

exclude = .gitignore, .git
ignore = E266,F841
```
