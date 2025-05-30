RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

FROM node

WORKDIR /app

COPY package*.json .

RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "start"]