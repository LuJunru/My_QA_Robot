# -*- coding: utf-8 -*-
# @Author  : Junru_Lu
# @File    : Online_Search_Module.py
# @Software: PyCharm
# @Environment : Python 3.6+

# 网页和服务请求相关包
from bs4 import BeautifulSoup
from urllib.parse import quote
import requests

# 基础包
import re
import os

# 编码相关包
import importlib, sys
importlib.reload(sys)

'''
本配置文件用于测试和编写在线搜索模块
'''

# ------------------预加载------------------ #

cur_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()  # 当前项目路径
    
photo_content = {}  # 百度知道上图片-文字对应表，目前共99个文字
for content_line in open(cur_dir + '/basic_data_file/photo_content.txt', 'r'):
    content = content_line.strip().split('\t')[1]
    photo_content[content] = content_line.strip().split('\t')[0]

ZHIDAOMAX = 1  # 控制在线搜索模块中从百度知道请求返回的问答数量。该值越大，返回的相似问答数量越多，但响应速度也就越慢


# ------------------在线搜索模块------------------ #

def get_html(url):  # 模拟代理请求和下载html
    headers = {'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}
    soup = BeautifulSoup(requests.get(url=url, headers=headers).content, "lxml")
    return soup


class OnlineSearch:  # 在线搜索：先在百度主页面上搜索，搜不到再到百度知道上搜索
    def __init__(self, text):
        self.content = text
        self.content1 = re.sub('天气', 'tianqi', self.content)
        self.main_url = 'https://www.baidu.com/s?wd=' + quote(self.content1)  # 百度知道上搜索的url请求
        self.main_soup = get_html(self.main_url)  # 百度主页面搜索结果返回的html

    def Baidu_Search(self):

        answer = {}
        try:
            result = self.main_soup.find(id=1)  # 获取html中第一个搜索结果
            d = result.attrs  # 检验搜索结果是否为空
        except:
            answer["安全回答"] = "我还在学习中，暂时无法回答您的问题"  # 无搜索结果时返回安全回答
            return answer

        # 给出知乎、搜狐或爱问知识人链接
        if result.find(class_='f13') and ("zhihu" in result.find(class_='f13').get_text()
                                          or "sohu" in result.find(class_='f13').get_text()
                                          or "iask" in result.find(class_='f13').get_text()):
                sublink = re.findall(r'[a-zA-z]+://[^\s]*', str(result))[0][:-1]
                answer[result.find(class_='t').get_text()] = sublink
                return answer

        if result.attrs.get('mu'):  # 检测搜索结果是否为百度知识图谱
            r = result.find(class_='op_exactqa_s_answer')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\n", "").replace(" ", "")
                return answer

        if result.attrs.get('mu') and \
                result.attrs['mu'].__contains__('http://open.baidu.com/static/calculator/calculator.html'):
            # 检测搜索结果是否为百度计算器
            r = result.find(class_='op_new_val_screen_result')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\xa0", "").replace(" ", "").replace("\n", "")
                return answer

        if result.attrs.get('tpl') and result.attrs['tpl'].__contains__('calendar'):
            # 检测搜索结果是否为日期(万年历)
            r = result.find(class_='op-calendar-content')
            if r is not None:
                answer['online answer'] = ''.join(re.compile('[\u4e00-\u9fa50-9\t]').
                    findall(re.sub('\s', '', str(r)).replace('</span>', '\t').replace('<span>', ''))[:-1])
                return answer

        try:
            result2 = self.main_soup.find(id=2)  # 获取html中第二个搜索结果
            d2 = result2.attrs  # 检验搜索结果是否为空
        except:
            result2 = '不存在'

        if result2 != '不存在':
            if result2.attrs.get('tpl') and result2.attrs['tpl'].__contains__('calendar'):
                # 检测搜索结果是否为日期(万年历)
                r = result2.find(class_='op-calendar-content')
                if r is not None:
                    answer['online answer'] = ''.join(re.compile('[\u4e00-\u9fa50-9\t]').\
                        findall(re.sub('\s', '', str(r)).replace('</span>', '\t').replace('<span>', ''))[:-1])
                    return answer

        if result.attrs.get('tpl') and "time" in result.attrs['tpl'] and "weather" not in result.attrs['tpl'] \
                and "news" not in result.attrs['tpl'] and "realtime" not in result.attrs['tpl']:
            # 检测搜索结果是否为日期或时间
            sublink = result.attrs['mu']
            if sublink == 'http://time.tianqi.com/':
                sublink = 'http://time.tianqi.com/beijing'
            r = get_html(sublink).find(class_='time').get_text()
            if r is not None:
                answer['online answer'] = r
                return answer

        if result.attrs.get('mu'):  # 检测搜索结果是否为百度天气
            r = result.find(class_='op_weather4_twoicon_today OP_LOG_LINK')
            if r is not None:
                answer['online answer'] = r.get_text().strip().replace("\n", "").replace(" ", "").replace('\xa0', '\n')
                return answer

        if result.attrs.get('tpl') and 'sp_fanyi' in result.attrs['tpl']:  # 是否为翻译
            r = result.find(class_='op_sp_fanyi_line_two')
            if r is not None:
                answer['online answer'] = r.get_text().strip()
                return answer

        if result.find("h3") is not None and result.find("h3").find("a").get_text().__contains__(u"百度百科"):
            # 检测搜索结果是否为百度百科
            url = result.find("h3").find("a")['href']
            if url is not None:
                baike_soup = get_html(url)  # 获取百度百科链接，进入百科，获取百科标题、摘要和基本信息
                r = baike_soup.find(class_='lemmaWgt-lemmaTitle lemmaWgt-lemmaTitle-')
                r1 = baike_soup.find(class_='lemma-summary')
                basicinfo = baike_soup.find_all("div", class_="basic-info cmn-clearfix")
                basicinfo2 = []  # 建立一个list存放最后的basicinfo
                for line in basicinfo:
                    i = 0
                    basicinfo_names = line.find_all("dt", class_="basicInfo-item name")  # 在basicinfo中获取全部basicinfo的项目名称
                    basicinfo_value = line.find_all("dd", class_="basicInfo-item value")  # 在basicinfo中获取全部basicinfo的项目内容
                    while i < len(basicinfo_names):  # 将basicinfo信息串联成一个字符串
                        basicinfo_value1 = re.sub('\r', '',
                                                  re.sub('\t', '', re.sub('\[\d\]', '', basicinfo_value[i].getText())))
                        basicinfo1 = basicinfo_names[i].getText() + ":" + basicinfo_value1
                        basicinfo2.append(basicinfo1)
                        i = i + 1
                if r1 is not None and r is not None:
                    r1 = r.get_text().strip().split('\n')[0] + ":" + r1.get_text().replace("\n", "").strip()
                    answer['百度百科'] = r1 + "\n" + "\n".join([re.subn('\n|\xa0', '', b)[0] for b in basicinfo2])
                    return answer

        if len(answer) == 0:  # 当百度主页面的第一条检索结果未能被上述条件捕获时，请求百度知道检索

            zhidao_url = "https://zhidao.baidu.com/search?word=" + quote(self.content)  # 百度主页面搜索的url请求
            zhidao_soup = get_html(zhidao_url)
            try:  # 一种情况是百度知道返回的搜索结果链接到了百度百科
                subsoup = get_html(zhidao_soup.find(class_='wgt-baike mb-20').find('a', href=True)['href'])
                r = subsoup.find(class_='lemmaWgt-lemmaTitle lemmaWgt-lemmaTitle-')
                r1 = subsoup.find(class_='lemma-summary')
                basicinfo = subsoup.find_all("div", class_="basic-info cmn-clearfix")
                basicinfo2 = []  # 建立一个list存放最后的basicinfo
                for line in basicinfo:
                    i = 0
                    basicinfo_names = line.find_all("dt", class_="basicInfo-item name")  # 在basicinfo中获取全部basicinfo的项目名称
                    basicinfo_value = line.find_all("dd",
                                                    class_="basicInfo-item value")  # 在basicinfo中获取全部basicinfo的项目内容
                    while i < len(basicinfo_names):  # 将basicinfo信息串联成一个字符串
                        basicinfo_value1 = re.sub('\r', '',
                                                  re.sub('\t', '', re.sub('\[\d\]', '', basicinfo_value[i].getText())))
                        basicinfo1 = basicinfo_names[i].getText() + ":" + basicinfo_value1
                        basicinfo2.append(basicinfo1)
                        i = i + 1
                if r1 is not None and r is not None:
                    r1 = r.get_text().strip().split('\n')[0] + ":" + r1.get_text().replace("\n", "").strip()
                    answer['百度百科'] = r1 + "\n" + "\n".join([re.subn('\n|\xa0', '', b)[0] for b in basicinfo2])
                    return answer
            except:
                try:  # 另一种情况是获取百度知道搜索结果中前三条带有最佳回答/推荐回答的问答
                    subsoups = [get_html(subsoup.find('a', href=True)['href']) for subsoup in zhidao_soup.find_all(class_='dt mb-8')]
                    if len(subsoups) == 0:
                        subsoups = [get_html(subsoup.find('a', href=True)['href']) for subsoup in zhidao_soup.find_all(class_='dt mb-4 line')[0:ZHIDAOMAX]]
                    p = 0
                    for subsoup in subsoups:
                        try:  # 在问答页面中，获取最佳回答，并解决文字以图片形式呈现和答案换行消失的问题
                            qtitle = subsoup.find(class_='ask-title').get_text().strip()
                            ans = subsoup.find(class_='bd answer').find('pre')
                            if ans is None:
                                ans = subsoup.find(class_='bd answer').find('ul')
                            if ans is None:
                                ans = subsoup.find(class_='bd answer').find('ol')
                            anss = re.sub("<br/>", "\n", str(ans))
                            anss = re.sub("<br>", "\n", anss)
                            anss = re.sub("<p/>", "\n", anss)
                            anss = re.sub("<p>", "\n", anss)
                            anss = re.sub("<li>", "·", anss)
                            anss = re.sub("·\n", "·", anss)
                            anss = re.sub("</li>", "\n", anss)
                            anss = re.sub("\n+", "\n", anss)
                            anss_list = [el.strip().split('"')[0] for el in re.findall(r'[a-zA-z]+://[^\s]*', anss)]
                            for eln in anss_list:
                                if photo_content.get(eln) is not None:
                                    ansss = anss.replace(eln, photo_content[eln])
                                    anss = ansss
                            anss = re.sub('<img class="word-replace" src="', '', anss).replace('"/>', '')
                            bdans = BeautifulSoup(anss, 'lxml').get_text().replace('\u3000', '').replace('查看原帖>>', '')
                            if bdans != 'None':
                                answer[qtitle] = bdans
                                p += 1
                        except:
                            pass
                except:
                    pass

        if len(answer) == 0:  # 实时搜索未能获取答案，返回安全回答
            answer["安全回答"] = "我还在学习中，暂时无法回答您的问题"

        return answer


# ------------------主函数------------------ #


if __name__ == '__main__':  # 供测试时使用

    while True:

        s1 = input("请输入查询的内容：")
        OS = OnlineSearch(s1)
        tub = OS.Baidu_Search()
        for tu in tub:
            print(tu, tub[tu])
        print('='*40)

    pass