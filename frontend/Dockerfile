# Website: build the React app, then serve it with nginx

# stage 1: build the static files (only this stage needs Node.js)
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# stage 2: a tiny web server with just the built files
FROM nginx:1.29-alpine
# files in templates/ get their ${VARIABLES} filled in at startup
COPY nginx.conf /etc/nginx/templates/default.conf.template
ENV BACKEND_URL=http://backend:8000
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
