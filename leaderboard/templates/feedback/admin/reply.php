<?php
/**
 * 页面功能：留言回复及删除的数据库操作实现
 */

session_start();

// 未登陆则重定向到登陆页面
if(!isset($_SESSION['username'])){
    header("Location: http://".$_SERVER['HTTP_HOST'].rtrim(dirname($_SERVER['PHP_SELF'] 
))."/login.php");
}

include '../conn.php';

if (isset($_POST['submit'])) {
	$repContent = htmlspecialchars(trim($_POST['reply']));
	$replytime = time();
	$update_sql = "update guestbook set reply='$repContent',replytime='$replytime' where id=".$_POST['id'];
	if (mysql_query($update_sql)) {
		 exit('<script language="javascript">alert("回复成功！");self.location="admin.php";</script>');
	}else{
		echo "回复失败……失败原因：".mysql_error()."<a href='javascript:history.back(-1);'>返回</a>";
	}
}elseif (isset($_GET['action']) && $_GET['action'] == 'delete') {
	$delete_sql = "delete from guestbook where id=".$_GET['id'];
	if (mysql_query($delete_sql)) {
		exit("<script language ='javascript'>alert('删除成功！');self:location='admin.php'</script>");
	}else{
		echo "删除失败,失败原因：".mysql_error()."<br/><a href='javascript:history.back(-1);>返回</a>'";
	}
	
}


