import requests
import urllib
from telegram.ext import CommandHandler, Updater, MessageHandler, Filters
from telegram import ReplyKeyboardMarkup
from sklearn.cluster import KMeans
from numpy import apply_along_axis, uint8
import numpy as np
from cv2 import imread, split, merge
from imageio import imwrite
from time import time
import warnings

warnings.simplefilter('ignore')

print('Start!')

token = "297826309:AAFd3cma2uayqEYxbipouh4MPrBRRav7Nro"
data = {}
text = "Этот бот может выполнить 3 действия: применить эффект постеризации, применить инверсию и выцветить изображение.\
Для того, чтобы выбрать режим работы, нужно написать команду /config и следовать дальнейшим инструкциям. При настройке\
постеризации вам будет задано несколько дополнительных вопросов. Во-первых, потребуется выбрать алгоритм постеризации.\
Обычно режим RGB_smart показывает лучшие результаты, но нельзя сказать, что он всегда работает лучше. Если вы не хотите\
разбираться в особенностях, то рекомендуется просто выбирать RGB_smart. Во-вторых, потребуется выбрать число цветов\
постеризации. Как следует из названия, этот параметр влияет на то, сколькими цветами будет заполнено результирующее\
изображение. Стоит сказать, что скорость выполнения запроса напрямую зависит от количества выбранных цветов постеризации: \
чем их меньше, тем быстрее придет ответ. Именно из-за этого максимум был ограничен 5 цветами, так как на обработку запроса\
с количеством цветов равным 10 уйдет около часа времени.\
\nДополнение: в будущем планируются расширить число возможных эффектов, добавить несколько фотофильтров, а также\
добавить несколько новых режимов работы постеризации."


def echo(bot, update):
    if update.message['photo']:
        mem = time()
        mode = data[update.message.chat_id][0]
        bot.sendMessage(update.message.chat.id, 'Обработка запроса может занять несколько минут, ожидайте')
        tmp = update.message['photo'][-1]
        id_ = tmp['file_id']
        del tmp
        url = requests.get('https://api.telegram.org/bot{}/getFile?file_id={}'.format(token, id_))
        path = url.json()['result']['file_path']
        url = 'https://api.telegram.org/file/bot{}/{}'.format(token, path)
        with open('local.jpg', 'wb') as file:
            file.write(urllib.request.urlopen(url).read())
        img = imread('local.jpg')
        b, g, r = split(img)
        img = merge([r, g, b])
        x, y = len(img), len(img[0])
        if mode == 'post':
            alg, k = data[update.message.chat_id][1:]
            transformed = img.reshape((x * y, 3))
            model = KMeans(n_clusters=k, init='k-means++' if alg[:-5] == 'smart' else 'random')
            y_pred = model.fit_predict(transformed)
            centers = model.cluster_centers_
            new_image = apply_along_axis(arr=y_pred, axis=0, func1d=lambda a: centers[a]).reshape(x, y, 3).astype(uint8)
            print('POST', str(k), str(x * y * 3), time() - mem)
        elif mode == 'invers':
            new_image = apply_along_axis(arr=img.flatten(), axis=0, func1d=lambda a: 255-a).reshape(x, y, 3).astype(uint8)
            print('INV', str(x * y * 3), time() - mem)
        elif mode == 'wb':
            new_image = np.repeat(np.round(np.sum(img * [.2989, .5870, .1140], axis=2)), 3, axis=1).astype(np.uint8).reshape((x, y, 3))
            print('WB', str(x * y * 3), time() - mem)
        imwrite('outfile.jpg', new_image)
        bot.sendPhoto(update.message.chat.id, open('outfile.jpg', 'rb'))
        print()
        #bot.sendMessage(update.message.chat.id, 'Score = {}'.format(str(score)))


def config(bot, update):
    reply_keyboard = [['/Posterisation', '/Inversion', '/WB']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text('Выберите режим обработки', reply_markup=markup)


def posterisation(bot, update):
    reply_keyboard = [['/RGB_smart', '/RGB']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text('Выберите алгоритм обработки (если не знаете, что выбрать, выбирайте RGB_smart)', reply_markup=markup)


def inversion(bot, update):
    data[update.message.chat_id] = ['invers']
    update.message.reply_text('Изменения сохранены!')


def wb(bot, update):
    data[update.message.chat_id] = ['wb']
    update.message.reply_text("Изменения сохранены!")


def rgb_smart(bot, update):
    data[update.message.chat_id] = ['post', 'RGB_smart']
    reply_keyboard = [['/2', '/3', '/5']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text("Выберите число цветов для постеризации", reply_markup=markup)


def rgb(bot, update):
    data[update.message.chat_id] = ['post', 'RGB']
    reply_keyboard = [['/2', '/3', '/5']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text("Выберите число цветов для постеризации", reply_markup=markup)


def upd(bot, update, n):
    if len(data[update.message.chat_id]) in [2, 3]:
        data[update.message.chat_id].append(n)
    else:
        update.message.reply_text("Ошибка")
        return
    update.message.reply_text("Изменения сохранены!")


def two(bot, update):
    upd(bot, update, 2)


def three(bot, update):
    upd(bot, update, 3)


def five(bot, update):
    upd(bot, update, 5)


def start(bot, update):
    data[update.message.chat_id] = []
    update.message.reply_text("Привет. Я - бот, который может наложить несколько эффектов на фотографию. Напиши /help, чтобы получить справку. Напиши /config, чтобы приступить к настройке")


def help_(bot, update):
    update.message.reply_text(text)


def main():
    updater = Updater(token)
    dp = updater.dispatcher
    text_handler = MessageHandler(Filters.photo, echo)
    dp.add_handler(text_handler)
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_))
    dp.add_handler(CommandHandler("config", config))
    dp.add_handler(CommandHandler("Posterisation", posterisation))
    dp.add_handler(CommandHandler("Inversion", inversion))
    dp.add_handler(CommandHandler("WB", wb))
    dp.add_handler(CommandHandler("RGB_smart", rgb_smart))
    dp.add_handler(CommandHandler("RGB", rgb))
    dp.add_handler(CommandHandler("2", two))
    dp.add_handler(CommandHandler("3", three))
    dp.add_handler(CommandHandler("5", five))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
