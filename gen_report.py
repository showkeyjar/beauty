# coding:utf-8
import webbrowser

"""
webbrowser.open(url, new=0, autoraise=True) 
Display url using the default browser. 
If new is 0, the url is opened in the same browser window if possible. 
If new is 1, a new browser window is opened if possible. 
If new is 2, a new browser page (“tab”) is opened if possible. 
If autoraise is True, the window is raised if possible 
(note that under many window managers this will occur regardless of the setting of this variable).
"""


def gen_report(filename, params, local=False):
    """
    生成颜值报告
    :param filename:
    :param params: [0,[],[]]
    :param local: 是否本地模式
    :return:
    """
    # 命名生成的html
    GEN_HTML = "templates/reports/" + filename + ".htm"
    # 打开文件，准备写入
    f = open(GEN_HTML, 'w', encoding='utf8')
    # 写入HTML界面中
    message = """
    <!DOCTYPE html>
    <html lang="cn">
    <head>
    <meta charset="UTF-8">
    <title>颜值报告</title>
    </head>
    <body>
    <h1>颜值报告</h1>
    <table width="100%%"><tr>
    <td width="50%%"><img src="%s" width="186" height="186" /></td>
    <td>
    <h2>颜值总得分：%s （总分5分）</h2>
    <p>您的优势：%s</p>
    <table><tr>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    </tr></table>
    <p>您的不足：%s</p>
    <table><tr>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    <td><img src="%s"></td>
    </tr></table>
    <p>相似美女：</p>
    
    <p>推荐改进：</p>
    
    </td>
    </tr></table>
    </body>
    </html>
    """ % tuple(params)
    # 写入文件
    f.write(message)
    # 关闭文件
    f.close()
    if local:
        # 运行完自动在网页中显示
        webbrowser.open(GEN_HTML, new=1)
