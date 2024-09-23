import random

def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case 401|403|404: #  多个case用"|"分隔
            return "Not allowed"
        case _:    # 设置默认case
            return "Something's wrong with the internet"

mystatus=random.randint(400,418)
print(mystatus)
print(http_error(mystatus))
# print(http_error(400))