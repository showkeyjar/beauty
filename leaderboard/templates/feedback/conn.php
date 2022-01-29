<?php
include 'config.php';

$conn = @mysql_connect(DB_HOST,DB_USER,DB_PASS);//如果成功 返回Mysql的连接标识 如果失败 返回false

if (!$conn) {
	exit('数据库连接失败'.mysql_error());
}

mysql_select_db('myguestbook',$conn);

mysql_query('set names utf8');