<?php
// 页面分为两部分：留言信息预处理和留言信息写入数据表

/*注意：Javascript 检测代码只是在当前页面友好的提醒用户将必须填写的信息填写完整，但不能保证提交到处理页面的信息也是完整的（如浏览器可以禁用 Javascript 代码而使之失效）。因此在处理表单信息的 PHP 程序里仍需对表单信息做检测。*/

 include_once 'conn.php';

 if (!isset($_POST['submit'])) {
 	exit('非法访问！');
 }

// 对系统参数get_magic_quotes_gpc()进行了检测 默认为开启装态 也可能是关闭状态 若为关闭状态 则进行addslashes进行转义处理

 if (get_magic_quotes_gpc()) {
	 $nickname = htmlspecialchars(trim($_POST['nickname']));
	 $email = isset($_POST['email'])?$_POST['email']:'';
	 $email = htmlspecialchars(trim($email));
	 $content = htmlspecialchars(trim($_POST['content']));

 }else{
 	$nickname = addslashes(htmlspecialchars(trim($_POST['nickname'])));
 	$email = isset($_POST['email'])?$_POST['email']:'';
 	$email = addslashes(htmlspecialchars(trim($email)));
 	$content = addslashes(htmlspecialchars(trim($_POST['content'])));
 }

 if (strlen($nickname) >16) {
 	exit("昵称长度不符合要求");
 }

 if ($email != '') {

 	if (strlen($email) > 60) {
 		exit("邮箱长度不符合要求");
 	}

	$pattern = "/^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$/";		
 	if (!preg_match($pattern, $email)) {
 		exit("邮箱格式不正确");
 	}
 }

$createtime = time();
$insert_sql = "insert into guestbook(nickname,email,content,createtime) values('$nickname','$email','$content','$createtime')";

$result_query = mysql_query($insert_sql);

if ($result_query) {
	
	// 由于写入成功后要使用html meta标签中的Refresh属性自动转至留言板界面 因此在两段php代码间加入了html代码
?>

	<!DOCTYPE html>
	<html lang="en">
	<head>
		<meta charset="UTF-8">
		<title>留言成功</title>
		<meta http-equiv="Refresh" content="2;url=index.php">
	</head>
	<body>
		<p>留言成功！非常感谢您的留言。<br/>请稍后，页面正在转回……</p>
	</body>
	</html>
	<?php
}else{
	exit("数据提交失败！".mysql_error()."<a href='javascript:history.back(-1);'>返回</a>");
	
}
?>

