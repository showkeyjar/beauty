<?php
// 登录检测 用户登录判断 如果没有登录而直接访问此页面 则重定向至login.php 
session_start();

if (!isset($_SESSION['id'])) {
	header("Location:login.php");
	exit();

}
?>

<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<title></title>
	<link rel="stylesheet" type="text/css" href="../main.css">
</head>
<body>

	<p><h3>留言内容</h3></p>
	<div class="comment">
		<?php
		// 展示留言数据 并作分页显示

		include_once '../conn.php';
		include_once '../config.php';

		$p = isset($_GET['p'])?$_GET['p']:1;

		$offset = ($p-1)*$pageSize;

		$sql_query = "select * from guestbook order by id desc limit $offset,$pageSize";
		$query_result = mysql_query($sql_query);
		if (!$query_result) {//对于select语句的查询 如果成功返回一个资源标识符 如果失败返回false
			exit("查询数据错误：".mysql_error());
		}


		while ($gb_array = mysql_fetch_array($query_result)) {
			$nickname = $gb_array['nickname'];
			$content = nl2br($gb_array['content']); //nl2br:在字符串中每个新行(\n)之前都插入HTML换行符<br/>
			$createtime = date('Y-m-d H:i',$gb_array['createtime']);
			echo $nickname." 发表于 ".$createtime."</br></br>";
			echo "内容：".$content."<br/><br/>";


			?>
			
			
			<div class="adminReply">
				<form action="reply.php" method="post" name="ReplyForm" class="ReplyForm">
						<p>
						<label for="reply" class="Replylabel">回复本条留言：</label>
						</p>
						<textarea name="reply" id="reply" cols="50" rows="3" class="replyText"><?=$gb_array['reply']?></textarea>
						<div class="ButtonDiv">
						<input type="hidden" name="id" value="<?=$gb_array['id'];?>">
						<input type="submit" name="submit" value="回复留言" class="button" />
						<span  class="button"><a href="reply.php?action=delete&id=<?php echo$gb_array['id'];?>">删除留言</a></span>
						</div>
				</form>
			</div>
			

			<?php
			echo "<hr/>";
		}

		$count_result = mysql_query("select count(*) from guestbook");
		$count_array = mysql_fetch_array($count_result);
		$pageNum = ceil($count_array['count(*)']/$pageSize);

		echo "共".$count_array['count(*)']."条留言：";	

		if ($pageNum>1) {
			for ($i=1; $i <= $pageNum; $i++) { 
				if ($i == $p) {
					echo '['.$i.']';
				}else{
					echo "&nbsp;<a href='admin.php?p=$i'>$i</a>&nbsp;";
				}
			}
		}
		?>
	</div>
</body>
</html>



