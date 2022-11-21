'''
[ 1. pre-commit 설치 ]
1. (생략)git init
2. pip install pre-commit                                  ## pre-commit 설치
3. pre-commit install                                      ## .pre-commit-config.yaml 파일을 기반으로 pre-commit 기능을 git에 설치: 최초 한번 실행
4. (생략:테스트용)pre-commit run                            ## pre-commit run --all-files
'''

'''
[ 2. pre-commit 실행 방법 ]
- 두 번 실행하기

1. git add .
2. git commit -m '[write the message]'
>> commit 실행시, config 파일을 통해 black, isort, autoflake8 실행되어 적용됨
>> 오류 및 변경 사항에 대해 출력 됨.

- 다시 git add와 commit을 실행하면 black,isort가 적용된 파일로 commit됨
1. git add .
2. git commit -m '[write the message]'
'''

'''
[ 3. 수정 사항 ]
- flake8 경우, setup.cfg파일에서 수정하기 > ignore에 무시할 오류 추가하기
'''
