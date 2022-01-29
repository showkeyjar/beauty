create database myguestbook;

use myguestbook;

create table guestbook(
	id mediumint(8) unsigned not null auto_increment key,
	nickname char(15) not null default 'guest',
	email varchar(30) not null default '',
	content text not null,
	createtime int(10) unsigned not null,
	reply text null,
	replytime int(10) unsigned
	)engine=myISAM default charset=utf8 auto_increment=1;

create table user(
id mediumint(15) unsigned not null auto_increment key,
username char(30) not null,
password varchar(50) not null
)engine=myISAM default charset=utf8 auto_increment=1;

insert into user(username,password) values('admin',md5('admin'));
 
insert into guestbook(id,nickname,email,content,createtime) values(1,'admin','admin@5idev.com','留言测试',126167501),(2,'user','user@163.com','大家好',1264169127),(3,'小明','xiaoming@163.com','做得好，大家努力',1264168865),
(4,'小张','xiaozhang@163.com','来看看',1264169118),(5,'小丽','xiaoli@163.com','哈哈',1283276566),
(6,'Tom','tom@gmail.com','hello',1283336218),(7,'Jack','jack@gmail.com','okok',1233366315),(8,'admin','admin@5idev.com','嗯嗯',1283337158);

