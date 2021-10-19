'''https://github.com/Cyp9715/Python-IMDB_review_crawler/blob/master/IMDB_crawler.py'''
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from multiprocessing import Pool, Value

# specify the location of file or variable, You don't have to do anything...
cromdriver_location = 'D:\py4e\ds\project_01\chromedriver_win32\chromedriver.exe'
tconstfile_location = 'D:\py4e\ds\project_01\imdb_crawl\Movie_code.tsv'
savefile_location = 'D:\py4e\ds\project_01\imdb_crawl'
errorfile_location = 'D:\py4e\ds\project_01\imdb_crawl\Error.csv'
multiprocess_count = 24

# 사용 변수
driver = webdriver
Star, User, Date, Title, Content = [], [], [], [], []


class Base:
    def tconst_list(self):
        global T_const
        T_const, T_const_Temp = [], []

        with open(tconstfile_location) as f:
            for line in f:
                T_const_Temp.append(line[:10])
            for i in range(T_const_Temp.__len__()):
                temp = str(T_const_Temp[i])
                temp = temp.replace('\n', '').replace("\"", '')
                T_const.append(temp)
            return T_const

    def init(self, temp):
        global counter
        counter = temp

    def plus_counter(self, number):
        global counter

        with counter.get_lock():
            counter.value += 1
            normal = (T_const[number] + " | " + str(counter.value) + "/" + str(T_const.__len__()) + " | " + str(
                round(counter.value / T_const.__len__() * 100, 3)) + "%")
            print(normal)


# Preprocessing 클래스가 인스턴스로 선언 될 때마다 __init__ 이 실행됩니다.
# tidied = [] 변수로 초기화 하는 과정을 거쳐야 trash_remove, trash_remove_Star의 동작이 따로 작동합니다,
# 만약 이 과정을 거치지 않으면 append에 Star,Title,Content 의 정보가 혼재되어 프로그램이 정상작동하지 않습니다.
class Preprocessing:
    def __init__(self):
        self.tidied = []

    def trash_remove(self, inputArray):
        for i in range(inputArray.__len__()):
            temp = str(inputArray[i])
            temp = temp.replace('**SPOILER ALERT**', '').replace('**Spoilers alert**', '').replace('\n', '')
            self.tidied.append(temp)
        return self.tidied

    def trash_remove_star(self, inputArray):
        for i in range(inputArray.__len__()):
            temp = str(inputArray[i])
            temp = temp.replace("/10", '')
            self.tidied.append(temp)
        return self.tidied


class Crawler:
    def click(self):
        global driver
        
        while True:
            try:
                time.sleep(5)
                driver.find_element_by_xpath("//*[@id='load-more-trigger']").click()
            except:
                break

    def find_web(self):
        global driver

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        reviews = soup.find_all('div', class_='review-container')

        for review in reviews:
            rating = review.find('span', class_='rating-other-user-rating')
            if rating:
                rating = ''.join(i.text for i in rating.find_all('span'))
            rating = rating if rating else '0'
            Star.append(rating)
            Title.append(review.find('a', class_='title').text.strip())
            User.append(review.find('span', class_='display-name-link').text)
            Date.append(review.find('span', class_='review-date').text)
            Content.append(review.find('div', class_='content').div.text)


def start(number):
    global Star
    global User
    global Date
    global Title
    global Content
    global driver
    global counter
    global T_const

    Craw = Crawler()

    # The [While True] is for resolving intermittent Chrome driver errors.
    while True:
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('headless')
            options.add_argument("disable-gpu")
            options.add_argument('--log-level=1')
            driver = webdriver.Chrome(executable_path=cromdriver_location, chrome_options=options)
            driver.get('https://www.imdb.com/title/' + str(T_const[number]) + '/reviews?ref_=tt_ql_3')

            Craw.click()
            Craw.find_web()
            Star = Preprocessing().trash_remove_star(Star)
            Title = Preprocessing().trash_remove(Title)
            Content = Preprocessing().trash_remove(Content)

            write = (savefile_location + str(T_const[number]) + ".tsv")
            with open(write, 'w', encoding='utf-8') as s:
                for row in range(Star.__len__()):
                    s.write(Star[row])
                    s.write("\t")
                    s.write(Date[row])
                    s.write("\t")
                    s.write(User[row])
                    s.write("\t")
                    s.write(Title[row])
                    s.write("\t")
                    s.write(Content[row])
                    s.write("\n")

            base.plus_counter(number)
            break

        except:
            print("Error!!" + T_const[number])
            write_error = errorfile_location
            with open(write_error, 'a', encoding='utf-8') as e:
                e.write(T_const[number])
                e.write("\n")

        finally:
            Star.clear()
            User.clear()
            Date.clear()
            Title.clear()
            Content.clear()
            driver.quit()


base = Base()
T_const = base.tconst_list()
counter = Value('i', 0)

if __name__ == '__main__':
    pool = Pool(initializer=base.init, initargs=(counter,), processes=multiprocess_count)
    pool.map(start, range(0, T_const.__len__()))